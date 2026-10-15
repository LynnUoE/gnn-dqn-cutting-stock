"""Simulated 2-D cutting stock environment (paper Sec. 2.3).

The paper fixes the action space, A = {(x, y, theta)} with |A| = 100 and
theta in {0, 90} degrees, and lists the cutting-state features, but does not
specify the stock, the orders or the reward. The choices made here:

* Stock sheet: 10 x 5 cells (a 2:1 panel), so X x Y x Theta = 10 x 5 x 2 = 100.
* Order: the sheet is split by random guillotine cuts into rectangles, which are
  then presented one at a time in random order. A waste-free layout therefore
  always exists and 100 % utilisation is the known optimum.
* Action: put the lower-left corner of the current piece at (x, y), rotated by theta.
* Reward: the placed area as a fraction of the sheet, so the return of an
  episode is its material utilisation; an infeasible cut costs one cell and
  the piece is discarded. A piece that fits nowhere on the remaining sheet is
  set aside without asking the agent, since no decision can place it.
* Material: directional fabrics have a grain, so their pieces may not be
  rotated. This is the only place where the texture class enters the dynamics
  and is what makes the graph embedding v_G informative for the agent.
"""

from __future__ import annotations

import numpy as np

ORIENTATIONS = (0, 90)
GRAINED_MATERIALS = frozenset({"corduroy", "cotton", "linen"})


class CuttingStockEnv:
    def __init__(
        self,
        width: int = 10,
        height: int = 5,
        max_piece_side: int = 4,
        stop_prob: float = 0.5,
        invalid_penalty: float = 0.02,
        grain_constraint: bool = True,
        seed: int | None = None,
    ):
        self.width = width
        self.height = height
        self.area = width * height
        self.max_piece_side = max_piece_side
        self.stop_prob = stop_prob
        self.invalid_penalty = invalid_penalty
        self.grain_constraint = grain_constraint
        self.rng = np.random.default_rng(seed)
        self.n_actions = self.area * len(ORIENTATIONS)
        self.feature_dim = 4 + self.area

    # ------------------------------------------------------------------ episode
    def reset(self, material: str | None = None) -> np.ndarray:
        self.grained = self.grain_constraint and material in GRAINED_MATERIALS
        self.occupied = np.zeros((self.height, self.width), dtype=bool)
        self.pieces = self._generate_order()
        self.cursor = 0
        self.cuts = 0
        self._skip_unplaceable()
        return self.features()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        valid = bool(self.valid_actions()[action])
        if valid:
            x, y, theta = self.decode(action)
            w, h = self.pieces[self.cursor]
            if theta == 90:
                w, h = h, w
            self.occupied[y : y + h, x : x + w] = True
            self.cuts += 1
            reward = w * h / self.area
        else:
            reward = -self.invalid_penalty
        self.cursor += 1
        self._skip_unplaceable()
        done = self.cursor >= len(self.pieces)
        return self.features(), reward, done, {"valid": valid, "utilization": self.utilization}

    # -------------------------------------------------------------------- state
    def features(self) -> np.ndarray:
        """f_C: remaining stock area, cuts made, current piece size, occupancy map."""
        w, h = self.pieces[self.cursor] if self.cursor < len(self.pieces) else (0, 0)
        head = [
            1.0 - self.utilization,
            self.cuts / self.area,
            w / self.max_piece_side,
            h / self.max_piece_side,
        ]
        return np.concatenate([head, self.occupied.ravel()]).astype(np.float32)

    @property
    def utilization(self) -> float:
        return float(self.occupied.sum()) / self.area

    # ------------------------------------------------------------------ actions
    def decode(self, action: int) -> tuple[int, int, int]:
        theta_idx, cell = divmod(int(action), self.area)
        y, x = divmod(cell, self.width)
        return x, y, ORIENTATIONS[theta_idx]

    def encode(self, x: int, y: int, theta: int) -> int:
        return ORIENTATIONS.index(theta) * self.area + y * self.width + x

    def valid_actions(self) -> np.ndarray:
        """Feasibility mask over the 100 actions for the current piece (all False once the order is done)."""
        mask = np.zeros((len(ORIENTATIONS), self.height, self.width), dtype=bool)
        if self.cursor < len(self.pieces):
            w, h = self.pieces[self.cursor]
            integral = np.pad(self.occupied, ((1, 0), (1, 0))).cumsum(axis=0).cumsum(axis=1)
            for i, theta in enumerate(ORIENTATIONS):
                pw, ph = (w, h) if theta == 0 else (h, w)
                if (theta == 90 and self.grained) or pw > self.width or ph > self.height:
                    continue
                covered = integral[ph:, pw:] - integral[:-ph, pw:] - integral[ph:, :-pw] + integral[:-ph, :-pw]
                mask[i, : self.height - ph + 1, : self.width - pw + 1] = covered == 0
        return mask.ravel()

    def first_fit_action(self) -> int:
        """Baseline heuristic: first free position in row-major order, unrotated first."""
        valid = np.flatnonzero(self.valid_actions())
        return int(min(valid, key=lambda a: (a % self.area, a // self.area)))

    # ------------------------------------------------------------------ helpers
    def _skip_unplaceable(self):
        while self.cursor < len(self.pieces) and not self.valid_actions().any():
            self.cursor += 1

    def _generate_order(self) -> list[tuple[int, int]]:
        todo = [(self.width, self.height)]
        pieces: list[tuple[int, int]] = []
        limit = self.max_piece_side
        while todo:
            w, h = todo.pop()
            oversized = w > limit or h > limit
            if not oversized and (w * h == 1 or self.rng.random() < self.stop_prob):
                pieces.append((w, h))
                continue
            axes = [i for i, side in enumerate((w, h)) if side > (limit if oversized else 1)]
            axis = int(self.rng.choice(axes))
            cut = int(self.rng.integers(1, (w, h)[axis]))
            if axis == 0:
                todo += [(cut, h), (w - cut, h)]
            else:
                todo += [(w, cut), (w, h - cut)]
        self.rng.shuffle(pieces)
        if not self.grained:
            # isotropic material: pieces arrive in arbitrary orientation
            pieces = [(h, w) if self.rng.random() < 0.5 else (w, h) for w, h in pieces]
        return [(int(w), int(h)) for w, h in pieces]
