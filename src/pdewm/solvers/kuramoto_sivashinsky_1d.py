from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from pdewm.solvers.base import BasePDESolver, SimulationResult
from pdewm.solvers.contexts import PDEContext


@dataclass(slots=True)
class KuramotoSivashinsky1DSolver(BasePDESolver):
    grid_size: int = 256
    domain_length: float = 22.0
    contour_points: int = 16
    dealias: bool = True
    linear_cfl: float = 0.5
    max_substeps_per_step: int = 4096
    solver_name: str = "ks_1d"

    def sample_initial_condition(self, rng: np.random.Generator, context: PDEContext) -> np.ndarray:
        amplitude = context.parameters.get("ic_amplitude", 1.0)
        bandwidth = int(context.parameters.get("ic_bandwidth", 8))
        domain_length = float(context.parameters.get("domain_length", self.domain_length))
        x = np.linspace(0.0, domain_length, self.grid_size, endpoint=False)
        state = np.zeros_like(x)

        for mode in range(1, bandwidth + 1):
            coefficient = rng.normal(scale=amplitude / (mode**1.5))
            phase = rng.uniform(0.0, 2.0 * np.pi)
            state += coefficient * np.cos(2.0 * np.pi * mode * x / domain_length + phase)

        return state[np.newaxis, :].astype(np.float32)

    def simulate(
        self,
        initial_state: np.ndarray,
        context: PDEContext,
        num_steps: int,
        dt: float | None = None,
        options: dict[str, Any] | None = None,
    ) -> SimulationResult:
        del options
        dt = float(dt or context.dt)
        domain_length = float(context.parameters.get("domain_length", self.domain_length))
        viscosity = float(context.parameters.get("viscosity", 1.0))
        warnings: list[str] = []

        u0 = np.asarray(initial_state, dtype=np.float64).reshape(self.grid_size)
        dx = domain_length / self.grid_size
        wave_numbers = 2.0 * np.pi * np.fft.fftfreq(self.grid_size, d=dx)
        linear_operator = wave_numbers**2 - viscosity * wave_numbers**4
        dealias_mask = self._dealias_mask(wave_numbers) if self.dealias else np.ones_like(wave_numbers)

        v = np.fft.fft(u0)
        trajectory = np.empty((num_steps + 1, 1, self.grid_size), dtype=np.float32)
        trajectory[0, 0] = u0.astype(np.float32)
        start = time.perf_counter()
        status = "ok"
        total_substeps = 0
        max_substeps_used = 0

        for step in range(1, num_steps + 1):
            dt_stable = self._stable_substep(linear_operator)
            substeps = int(np.ceil(dt / max(dt_stable, 1.0e-12)))
            substeps = max(1, min(substeps, self.max_substeps_per_step))
            dt_sub = dt / substeps
            E, E2, Q, f1, f2, f3 = self._etdrk4_coefficients(linear_operator, dt_sub)
            for _ in range(substeps):
                nonlinear_v = self._nonlinear_term(v, wave_numbers, dealias_mask)
                a = E2 * v + Q * nonlinear_v
                nonlinear_a = self._nonlinear_term(a, wave_numbers, dealias_mask)
                b = E2 * v + Q * nonlinear_a
                nonlinear_b = self._nonlinear_term(b, wave_numbers, dealias_mask)
                c = E2 * a + Q * (2.0 * nonlinear_b - nonlinear_v)
                nonlinear_c = self._nonlinear_term(c, wave_numbers, dealias_mask)
                v = E * v + f1 * nonlinear_v + 2.0 * f2 * (nonlinear_a + nonlinear_b) + f3 * nonlinear_c
                total_substeps += 1

            max_substeps_used = max(max_substeps_used, substeps)
            state = np.fft.ifft(v).real

            if not np.isfinite(state).all():
                warnings.append(f"non-finite state detected at step {step}")
                status = "nan"
                trajectory = trajectory[:step]
                break

            if float(np.max(np.abs(state))) > 1.0e5:
                warnings.append(f"instability threshold reached at step {step}")
                status = "unstable"
                trajectory = trajectory[:step]
                break

            trajectory[step, 0] = state.astype(np.float32)

        runtime_sec = time.perf_counter() - start
        diagnostics = {
            "max_abs_value": float(np.max(np.abs(trajectory))),
            "mean_energy": float(np.mean(trajectory**2)),
            "steps_completed": int(trajectory.shape[0] - 1),
            "total_substeps": int(total_substeps),
            "max_substeps_used": int(max_substeps_used),
            "viscosity": float(viscosity),
        }
        return SimulationResult(
            trajectory=trajectory,
            status=status,
            runtime_sec=runtime_sec,
            warnings=warnings,
            diagnostics=diagnostics,
        )

    def _dealias_mask(self, wave_numbers: np.ndarray) -> np.ndarray:
        cutoff = (2.0 / 3.0) * np.max(np.abs(wave_numbers))
        return (np.abs(wave_numbers) <= cutoff).astype(np.float64)

    def _nonlinear_term(
        self,
        v: np.ndarray,
        wave_numbers: np.ndarray,
        dealias_mask: np.ndarray,
    ) -> np.ndarray:
        state = np.fft.ifft(v).real
        nonlinear_hat = -0.5j * wave_numbers * np.fft.fft(state**2)
        return nonlinear_hat * dealias_mask

    def _etdrk4_coefficients(
        self,
        linear_operator: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        lr_roots = np.exp(
            1j * np.pi * (np.arange(1, self.contour_points + 1) - 0.5) / self.contour_points
        )
        lr = dt * linear_operator[:, None] + lr_roots[None, :]
        E = np.exp(dt * linear_operator)
        E2 = np.exp(0.5 * dt * linear_operator)
        Q = dt * np.real(np.mean((np.exp(lr / 2.0) - 1.0) / lr, axis=1))
        f1 = dt * np.real(
            np.mean((-4.0 - lr + np.exp(lr) * (4.0 - 3.0 * lr + lr**2)) / lr**3, axis=1)
        )
        f2 = dt * np.real(
            np.mean((2.0 + lr + np.exp(lr) * (-2.0 + lr)) / lr**3, axis=1)
        )
        f3 = dt * np.real(
            np.mean((-4.0 - 3.0 * lr - lr**2 + np.exp(lr) * (4.0 - lr)) / lr**3, axis=1)
        )
        return E, E2, Q, f1, f2, f3

    def _stable_substep(self, linear_operator: np.ndarray) -> float:
        max_rate = float(np.max(np.abs(linear_operator)))
        if max_rate <= 0.0:
            return np.inf
        return float(self.linear_cfl / max_rate)

