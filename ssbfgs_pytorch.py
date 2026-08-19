"""GPU-capable self-scaled BFGS and Broyden optimizers for PyTorch.

The update equations follow the project's modified SciPy ``_optimize.py`` and
the SSBFGS branch of raj-brown/optimistix. Unlike the SciPy implementation, all
vectors and the inverse-Hessian approximation are torch tensors and remain on
the device of the optimized parameters.

These are full-memory methods: the inverse Hessian requires O(n**2) storage.
They are consequently intended for small/medium models or for optimizing a
small subset of a larger model.
"""

from __future__ import annotations

from typing import Callable, Iterable, Tuple

import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lbfgs import _strong_wolfe

Closure = Callable[[], Tensor]


class _SelfScaledQN(Optimizer):
    """Shared machinery for dense, self-scaled quasi-Newton methods."""

    def __init__(
        self,
        params: Iterable[Tensor],
        *,
        lr: float = 1.0,
        max_iter: int = 20,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        line_search: str | bool = "armijo",
        c1: float = 1e-4,
        c2: float = 0.9,
        backtrack: float = 0.5,
        max_line_search: int = 25,
        initial_scale: bool = False,
        curvature_eps: float = 1e-12,
    ) -> None:
        if lr <= 0:
            raise ValueError("lr must be positive")
        if max_iter < 1 or max_line_search < 1:
            raise ValueError("max_iter and max_line_search must be positive")
        if not 0.0 < c1 < 1.0:
            raise ValueError("c1 must lie in (0, 1)")
        if not c1 < c2 < 1.0:
            raise ValueError("c2 must lie in (c1, 1)")
        if not 0.0 < backtrack < 1.0:
            raise ValueError("backtrack must lie in (0, 1)")
        if curvature_eps <= 0:
            raise ValueError("curvature_eps must be positive")

        if isinstance(line_search, bool):
            line_search = "armijo" if line_search else "none"
        line_search = line_search.lower()
        if line_search not in {"none", "armijo", "strong_wolfe"}:
            raise ValueError("line_search must be 'none', 'armijo', or 'strong_wolfe'")

        defaults = dict(
            lr=float(lr),
            max_iter=int(max_iter),
            tolerance_grad=float(tolerance_grad),
            tolerance_change=float(tolerance_change),
            line_search=line_search,
            c1=float(c1),
            c2=float(c2),
            backtrack=float(backtrack),
            max_line_search=int(max_line_search),
            initial_scale=bool(initial_scale),
            curvature_eps=float(curvature_eps),
        )
        super().__init__(params, defaults)

        if len(self.param_groups) != 1:
            raise ValueError("self-scaled quasi-Newton optimizers support one parameter group")
        self._params = list(self.param_groups[0]["params"])
        if not self._params:
            raise ValueError("optimizer received an empty parameter list")
        device = self._params[0].device
        dtype = self._params[0].dtype
        if not (dtype.is_floating_point or dtype.is_complex):
            raise TypeError("parameters must have a floating-point dtype")
        if any(p.device != device or p.dtype != dtype for p in self._params):
            raise ValueError("all parameters must have the same device and dtype")
        if any(p.is_complex() for p in self._params):
            raise TypeError("complex parameters are not supported")

        self._numel = sum(p.numel() for p in self._params)
        # Keep global optimizer state under the first parameter, as LBFGS does.
        self.state[self._params[0]]["H"] = torch.eye(
            self._numel, device=device, dtype=dtype
        )
        self.state[self._params[0]]["n_iter"] = 0
        self.state[self._params[0]]["func_evals"] = 0
        # Objective at the point preceding the current iterate.  SciPy's
        # Wolfe search uses this to choose its first trial step.
        self.state[self._params[0]]["old_old_loss"] = None

    def _gather_flat_grad(self) -> Tensor:
        views = []
        for p in self._params:
            if p.grad is None:
                views.append(torch.zeros_like(p, memory_format=torch.contiguous_format).view(-1))
            elif p.grad.is_sparse:
                views.append(p.grad.to_dense().reshape(-1))
            else:
                views.append(p.grad.reshape(-1))
        return torch.cat(views).detach()

    @torch.no_grad()
    def _gather_flat_params(self) -> Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self._params])

    @torch.no_grad()
    def _set_flat_params(self, flat: Tensor) -> None:
        offset = 0
        for p in self._params:
            count = p.numel()
            p.copy_(flat[offset : offset + count].view_as(p))
            offset += count

    def _evaluate(self, closure: Closure) -> Tuple[Tensor, Tensor]:
        with torch.enable_grad():
            loss = closure()
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise ValueError("closure must return a scalar torch.Tensor")
        if loss.device != self._params[0].device:
            raise ValueError("loss and optimized parameters must be on the same device")
        self.state[self._params[0]]["func_evals"] += 1
        return loss.detach(), self._gather_flat_grad()

    def _take_step(
        self,
        closure: Closure,
        x: Tensor,
        loss: Tensor,
        grad: Tensor,
        direction: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, float]:
        cfg = self.param_groups[0]
        state = self.state[self._params[0]]
        directional_derivative = torch.dot(grad, direction)
        alpha = cfg["lr"]

        # Match scalar_search_wolfe1's initial-step estimate.  On the first
        # iteration SciPy defines old_old_fval = f + ||g||/2; afterward it
        # carries the objective from the preceding iterate.
        if cfg["line_search"] == "strong_wolfe":
            old_old_loss = state["old_old_loss"]
            if old_old_loss is None:
                old_old_loss = float(loss.item()) + float(torch.linalg.vector_norm(grad).item()) / 2.0
            slope = float(directional_derivative.item())
            if slope != 0.0:
                trial = 1.01 * 2.0 * (float(loss.item()) - old_old_loss) / slope
                if trial >= 0.0:
                    alpha = min(alpha, trial)

        if cfg["line_search"] == "none":
            candidate = x + alpha * direction
            self._set_flat_params(candidate)
            new_loss, new_grad = self._evaluate(closure)
            return candidate, new_loss, new_grad, alpha

        if cfg["line_search"] == "strong_wolfe":
            def objective_at_step(_x, step, search_direction):
                self._set_flat_params(x + step * search_direction)
                trial_loss, trial_grad = self._evaluate(closure)
                return float(trial_loss.item()), trial_grad

            new_loss_float, new_grad, alpha, _ = _strong_wolfe(
                objective_at_step,
                x,
                alpha,
                direction,
                float(loss.item()),
                grad,
                directional_derivative,
                c1=cfg["c1"],
                c2=cfg["c2"],
                # Do not impose a 1e-14 step/bracket floor on float64 runs;
                # it can terminate the search before objectives in the
                # 1e-13--1e-14 range have converged.  Machine epsilon is the
                # smallest meaningful positive tolerance for this dtype.
                tolerance_change=max(
                    cfg["tolerance_change"], torch.finfo(x.dtype).eps
                ),
                max_ls=cfg["max_line_search"],
            )
            candidate = x + alpha * direction
            self._set_flat_params(candidate)
            new_loss = loss.new_tensor(new_loss_float)

            # torch's private routine returns its best bracket point when it
            # exhausts the search, even if that point does not satisfy strong
            # Wolfe.  _line_search_wolfe12 instead reports failure.  Validate
            # both conditions explicitly and restore x before raising.
            new_slope = torch.dot(new_grad, direction)
            armijo_ok = new_loss <= loss + cfg["c1"] * alpha * directional_derivative
            curvature_ok = new_slope.abs() <= -cfg["c2"] * directional_derivative
            if not (
                alpha > 0.0
                and torch.isfinite(new_loss).item()
                and torch.isfinite(new_grad).all().item()
                and armijo_ok.item()
                and curvature_ok.item()
            ):
                self._set_flat_params(x)
                raise RuntimeError(
                    "strong-Wolfe line search failed to satisfy the Armijo "
                    "and curvature conditions"
                )
            return candidate, new_loss, new_grad, alpha

        # Armijo backtracking. Objective/gradient work is still done on the
        # model's device; scalar comparisons necessarily synchronize CUDA.
        for _ in range(cfg["max_line_search"]):
            candidate = x + alpha * direction
            self._set_flat_params(candidate)
            new_loss, new_grad = self._evaluate(closure)
            armijo_rhs = loss + cfg["c1"] * alpha * directional_derivative
            if torch.isfinite(new_loss).item() and (new_loss <= armijo_rhs).item():
                return candidate, new_loss, new_grad, alpha
            alpha *= cfg["backtrack"]

        # A failed line search must leave the model at the original point.
        self._set_flat_params(x)
        raise RuntimeError(
            "line search failed; try a smaller lr, a larger max_line_search, "
            "or double precision"
        )

    @staticmethod
    def _safe_nonzero(value: Tensor, eps: float) -> bool:
        return bool(torch.isfinite(value).item() and (value.abs() > eps).item())

    def _updated_inverse_hessian(
        self,
        H: Tensor,
        s: Tensor,
        y: Tensor,
        old_grad: Tensor,
        new_grad: Tensor,
        alpha: float,
        first_update: bool,
    ) -> Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def step(self, closure: Closure) -> Tensor:
        """Run up to ``max_iter`` quasi-Newton iterations.

        The closure must clear gradients, compute the scalar objective, call
        ``backward()``, and return the objective.  The returned tensor is the
        final objective value (unlike PyTorch LBFGS, which returns the initial
        value).
        """
        if closure is None:
            raise ValueError("a closure is required")

        cfg = self.param_groups[0]
        state = self.state[self._params[0]]
        loss, grad = self._evaluate(closure)
        H = state["H"]

        for _ in range(cfg["max_iter"]):
            if not torch.isfinite(loss).item() or not torch.isfinite(grad).all().item():
                raise FloatingPointError("non-finite objective or gradient")
            if grad.abs().max().item() <= cfg["tolerance_grad"]:
                break

            x = self._gather_flat_params()
            direction = -(H @ grad)
            slope = torch.dot(grad, direction)
            # Numerical error or an unsafe update can destroy descent. Reset
            # to steepest descent and restart the inverse Hessian in that case.
            if not torch.isfinite(direction).all().item() or slope.item() >= 0.0:
                H = torch.eye(self._numel, device=x.device, dtype=x.dtype)
                direction = -grad

            new_x, new_loss, new_grad, alpha = self._take_step(
                closure, x, loss, grad, direction
            )
            s = new_x - x
            y = new_grad - grad
            step_small = s.abs().max().item() <= cfg["tolerance_change"]
            loss_small = abs((new_loss - loss).item()) <= cfg["tolerance_change"]

            ys = torch.dot(y, s)
            curvature_floor = cfg["curvature_eps"] * max(
                1.0, (torch.linalg.vector_norm(y) * torch.linalg.vector_norm(s)).item()
            )
            if torch.isfinite(ys).item() and ys.item() > curvature_floor:
                candidate_H = self._updated_inverse_hessian(
                    H, s, y, grad, new_grad, alpha, state["n_iter"] == 0
                )
                if torch.isfinite(candidate_H).all().item():
                    # Roundoff can slowly break symmetry.
                    H = 0.5 * (candidate_H + candidate_H.T)

            # Preserve the objective at x for SciPy's next initial-step guess.
            state["old_old_loss"] = float(loss.item())
            loss, grad = new_loss, new_grad
            state["n_iter"] += 1
            if step_small or loss_small:
                # A line search can legally return a step so small that the
                # parameters do not change at machine precision.  Retaining
                # the same quasi-Newton matrix then reproduces that zero step
                # on every subsequent ``step`` call (the Helmholtz SSBFGS run
                # previously stalled this way).  Restart from steepest
                # descent on the next call instead.
                H = torch.eye(self._numel, device=new_x.device, dtype=new_x.dtype)
                break

        state["H"] = H
        return loss


class SSBFGS(_SelfScaledQN):
    """Self-scaled BFGS with the Oren-Luenberger or Al-Baali scale.

    Args:
        variant: ``"OL"`` uses ``tau = 1 / b``; ``"AB"`` uses
            ``tau = min(1, 1 / b)``.
        initial_scale: Apply the usual ``(y.T y)/(y.T s)`` scale on the first
            update. This matches the option in the modified SciPy code.
        Other keyword arguments are documented by :class:`_SelfScaledQN`.
    """

    def __init__(self, params: Iterable[Tensor], *, variant: str = "AB", **kwargs) -> None:
        variant = variant.upper()
        if variant not in {"OL", "AB"}:
            raise ValueError("SSBFGS variant must be 'OL' or 'AB'")
        self.variant = variant
        kwargs.setdefault("line_search", "strong_wolfe")
        super().__init__(params, **kwargs)

    def _updated_inverse_hessian(
        self, H, s, y, old_grad, new_grad, alpha, first_update
    ) -> Tensor:
        eps = self.param_groups[0]["curvature_eps"]
        ys = torch.dot(y, s)
        rho = ys.reciprocal()

        if first_update and self.param_groups[0]["initial_scale"]:
            tau = rho * torch.dot(y, y)
        else:
            # gfkp1 - yk == gfk in the original implementation.
            b = -alpha * rho * torch.dot(s, old_grad)
            if not self._safe_nonzero(b, eps) or b.item() <= 0.0:
                return H
            tau = b.reciprocal()
            if self.variant == "AB":
                tau = torch.minimum(tau, tau.new_tensor(1.0))

        if not self._safe_nonzero(tau, eps) or tau.item() <= 0.0:
            return H
        Hs = H / tau
        Hy = Hs @ y
        yHy = torch.dot(y, Hy)
        return (
            Hs
            - rho * (torch.outer(Hy, s) + torch.outer(s, Hy))
            + rho * (1.0 + rho * yHy) * torch.outer(s, s)
        )


class SSBROYDEN(_SelfScaledQN):
    """Self-scaled Broyden-family inverse-Hessian optimizer.

    ``variant=1`` selects the SR1-clipped theta rule (``SSBroyden1`` in the
    modified SciPy code); ``variant=2`` selects its direct clipped-theta rule.
    """

    def __init__(self, params: Iterable[Tensor], *, variant: int = 1, **kwargs) -> None:
        if variant not in {1, 2}:
            raise ValueError("SSBROYDEN variant must be 1 or 2")
        self.variant = variant
        # ``_minimize_bfgs`` in the project's _optimize.py applies the same
        # strong-Wolfe line search to BFGS and both Broyden variants.
        kwargs.setdefault("line_search", "strong_wolfe")
        super().__init__(params, **kwargs)

    def _updated_inverse_hessian(
        self, H, s, y, old_grad, new_grad, alpha, first_update
    ) -> Tensor:
        cfg = self.param_groups[0]
        eps = cfg["curvature_eps"]
        n = s.numel()
        if n <= 1:
            # The Broyden self-scale contains exponent 1/(1-n).
            ys = torch.dot(y, s)
            return torch.outer(s, s) / ys

        ys = torch.dot(y, s)
        rho = ys.reciprocal()
        Hy = H @ y
        yHy = torch.dot(y, Hy)
        if not self._safe_nonzero(yHy, eps) or yHy.item() <= 0.0:
            return H

        h = yHy * rho
        b = -alpha * rho * torch.dot(s, old_grad)
        if not self._safe_nonzero(b, eps) or b.item() <= 0.0:
            return H
        # Match _optimize.py exactly: a_k = h_k b_k - 1.  Only the numerator
        # inside the square root uses |a_k|; replacing a_k itself by |a_k|
        # changes theta, sigma, phi, and ultimately the Hessian update.
        a = b * h - 1.0
        one_plus_a = 1.0 + a
        if not self._safe_nonzero(one_plus_a, eps) or one_plus_a.item() <= 0.0:
            return H

        rho_m = torch.minimum(
            h.new_tensor(1.0),
            h * (1.0 - torch.sqrt(torch.abs(a) / one_plus_a)),
        )
        if not self._safe_nonzero(a, eps) or not self._safe_nonzero(rho_m, eps):
            return H
        theta_m = (rho_m - 1.0) / a
        theta_p = rho_m.reciprocal()

        if self.variant == 1:
            one_minus_b = 1.0 - b
            if self._safe_nonzero(one_minus_b, eps):
                theta_sr1 = one_minus_b.reciprocal()
                if b.item() > 1.0:
                    theta = torch.maximum(theta_m, theta_sr1)
                else:
                    theta = torch.minimum(theta_p, theta_sr1)
            else:
                theta = theta_m
        else:
            theta = torch.maximum(theta_m, torch.minimum(theta_p, (1.0 - b) / b))

        sigma = 1.0 + theta * a
        if not self._safe_nonzero(sigma, eps):
            return H

        if first_update and cfg["initial_scale"]:
            tau = h / sigma
        else:
            rho_k = torch.minimum(b.new_tensor(1.0), b.reciprocal())
            sigma_power = torch.abs(sigma).pow(1.0 / (1.0 - n))
            if theta.item() <= 0.0:
                tau = torch.minimum(rho_k * sigma_power, sigma)
            else:
                if not self._safe_nonzero(theta, eps):
                    return H
                tau = rho_k * torch.minimum(sigma_power, theta.reciprocal())

        theta_denom = 1.0 + a * theta
        if (
            not self._safe_nonzero(tau, eps)
            or tau.item() <= 0.0
            or not self._safe_nonzero(theta_denom, eps)
        ):
            return H
        v = s * rho - Hy / yHy
        phi = (1.0 - theta) / theta_denom
        return (
            H - torch.outer(Hy, Hy) / yHy + phi * yHy * torch.outer(v, v)
        ) / tau + rho * torch.outer(s, s)


# A conventional spelling is useful in imports while retaining the requested
# all-caps optimizer name.
SSBroyden = SSBROYDEN

__all__ = ["SSBFGS", "SSBROYDEN", "SSBroyden"]
