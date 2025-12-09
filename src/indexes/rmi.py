"""
===============================================================================
TWO-STAGE RECURSIVE MODEL INDEX (RMI)
===============================================================================
A compact two-stage learned index:
  -Stage 0 (root): a single linear model that predicts the "position" of a key.
  -Stage 1 (leaves): a set of per-segment linear models that refine the position.

Build:
  1) Fit a root linear regression mapping key -> global position [0..n-1].
  2) Partition the "positions" into `fanout` contiguous segments of (roughly) equal
     size, and for each segment fit a linear model mapping key -> global position.
  3) For each leaf, compute the max absolute training error, store it as the
     segment-specific search window (plus a small safety buffer at query time).

Search(key):
  -Use root model to predict a global position, convert to a segment id.
  -Use the leaf model for that segment to predict a refined position.
  -Binary-search only within [pred - window, pred + window] intersected with the
    segment's [start, end) bounds.

===============================================================================
"""

import bisect
import numpy as np
from typing import Tuple


class RecursiveModelIndex:
    """Two-stage Recursive Model Index (RMI) with linear models at both stages."""

    def __init__(self, fanout: int = 128):
        """
        Args:
            fanout: Number of leaf models (segments). Typical: 128-8192 depending
                    on dataset size and desired accuracy.
        """
        self.fanout = int(max(1, fanout))

        # Trained artifacts
        self.keys: np.ndarray | None = None  # Sorted keys
        self.n: int = 0

        # Stage 0 (root) linear model: pos ≈ a0 * key + b0
        self.a0: float = 0.0
        self.b0: float = 0.0

        # Stage 1 (leaf) models per segment: pos ≈ a[s] * key + b[s]
        self.seg_a: np.ndarray | None = None 
        self.seg_b: np.ndarray | None = None

        # Segment bounds in *global* position space [start, end)
        self.seg_start: np.ndarray | None = None
        self.seg_end: np.ndarray | None = None 

        # Training max-absolute-error per segment (int indices)
        self.seg_err: np.ndarray | None = None 

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build_from_sorted_array(self, keys: np.ndarray):
        """Fit the two-stage RMI from sorted keys.

        Args:
            keys: sorted 1-D NumPy array of keys (float or int). Length can be 0.
        """
        # Normalize input
        if keys is None:
            keys = np.array([], dtype=np.float64)
        self.keys = np.asarray(keys)
        self.n = int(self.keys.shape[0])

        # Handle trivial cases
        if self.n == 0:
            # Leave defaults, searches will immediately fail gracefully.
            self._alloc_segments(0)
            return
        if self.n == 1:
            # Degenerate: any key equal to keys[0] is found at pos 0.
            self._alloc_segments(self.fanout)
            self.a0, self.b0 = 0.0, 0.0
            self.seg_start[:] = 0
            self.seg_end[:] = 1
            self.seg_a[:] = 0.0
            self.seg_b[:] = 0.0
            self.seg_err[:] = 0
            return

        # Fit root model mapping key -> global position in [0, n-1]
        positions = np.arange(self.n, dtype=np.float64)
        self.a0, self.b0 = np.polyfit(self.keys, positions, 1)

        # Partition the "positions" into 'fanout' contiguous segments of equal size
        self._alloc_segments(self.fanout)
        for s in range(self.fanout):
            start = int(np.floor(s * self.n / self.fanout))
            end = int(np.floor((s + 1) * self.n / self.fanout))
            self.seg_start[s] = start
            self.seg_end[s] = end

        # Fit per-segment models and record training errors
        for s in range(self.fanout):
            start, end = int(self.seg_start[s]), int(self.seg_end[s])
            if end <= start:
                # Empty segment
                self.seg_a[s] = 0.0
                self.seg_b[s] = float(start)
                self.seg_err[s] = 0
                continue

            kseg = self.keys[start:end]
            pseg = positions[start:end]

            # If all keys are identical, fall back
            if kseg.size < 2 or np.all(kseg == kseg[0]):
                self.seg_a[s] = 0.0
                self.seg_b[s] = float(start)
                # Error bound is segment length (worst-case) to be safe
                self.seg_err[s] = int(end - start)
                continue

            # Fit linear model pos ≈ a*s + b for this segment
            a, b = np.polyfit(kseg, pseg, 1)
            self.seg_a[s] = float(a)
            self.seg_b[s] = float(b)

            # Compute max absolute training error for this segment
            pred = a * kseg + b
            max_err = int(np.max(np.abs(pred - pseg))) if kseg.size else 0
            # Keep a reasonable minimum window to absorb numeric/bisect effects
            self.seg_err[s] = int(max(2, max_err))

    def _alloc_segments(self, fanout: int):
        f = int(max(1, fanout))
        self.seg_a = np.zeros(f, dtype=np.float64)
        self.seg_b = np.zeros(f, dtype=np.float64)
        self.seg_start = np.zeros(f, dtype=np.int64)
        self.seg_end = np.zeros(f, dtype=np.int64)
        self.seg_err = np.zeros(f, dtype=np.int64)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    def search(self, key: float, safety: int = 8) -> bool:
        """Predict the key's position using the two-stage model and correct locally.

        Args:
            key: query key
            safety: additional window (indices) added to the learned error bound
                    per segment to compensate for distribution drift.
        Returns:
            bool indicating whether the key was found.
        """
        if self.keys is None or self.n == 0:
            return False
        if self.n == 1:
            return bool(key == self.keys[0])

        # Stage 0: predict global position and map to segment id
        pos0 = self.a0 * key + self.b0
        # Convert pos to segment id using equal-sized position partitions
        denom = max(1.0, float(self.n - 1))
        s = int(np.floor(np.clip((pos0 / denom) * self.fanout, 0, self.fanout - 1)))

        # Stage 1: leaf model for segment s
        a = float(self.seg_a[s])
        b = float(self.seg_b[s])
        pred = int(np.round(a * key + b))

        # Window based on training error + safety margin
        w = int(self.seg_err[s]) + int(max(0, safety))

        # Clamp to segment bounds
        left = max(int(self.seg_start[s]), pred - w)
        right = min(int(self.seg_end[s]), pred + w + 1)
        if right <= left:
            left = int(self.seg_start[s])
            right = int(self.seg_end[s])

        # Local binary search within [left, right)
        view = self.keys[left:right]
        idx = bisect.bisect_left(view, key)
        found = (idx + left < self.n) and (self.keys[idx + left] == key)
        return found

    # ------------------------------------------------------------------
    # Memory usage estimate
    # ------------------------------------------------------------------
    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes for the model + keys (if stored)."""
        total = 0
        if self.keys is not None:
            total += self.keys.nbytes
        # Root params
        total += 16  # a0, b0 (~2*8 bytes)
        # Leaf params and metadata
        for arr in (self.seg_a, self.seg_b):
            if arr is not None:
                total += arr.nbytes
        for arr in (self.seg_start, self.seg_end, self.seg_err):
            if arr is not None:
                total += arr.nbytes
        # Object overhead fudge factor
        total += 256
        return int(total)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------ 
    def predict(self, key: float) -> float:
        if self.keys is None or self.n == 0:
            return -1

        # Stage 0: predict global position and map to segment id
        pos0 = self.a0 * key + self.b0

        # Convert pos to segment id using equal-sized position partitions
        denom = max(1.0, float(self.n - 1))
        s = int(np.floor(np.clip((pos0 / denom) * self.fanout, 0, self.fanout - 1)))

        # Stage 1: Leaf Model
        return float(self.seg_a[s] * key + self.seg_b[s])

# Quick self-test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # Build a moderately large sorted key array with some skew
    u = np.sort(rng.uniform(0, 1_000_000, 40_000))
    rmi = RecursiveModelIndex(fanout=128)
    rmi.build_from_sorted_array(u)

    # Hit rate on in-distribution queries
    queries = np.concatenate([
        rng.choice(u, 5_000, replace=True),
        rng.uniform(u.min(), u.max(), 5_000),
    ])
    hits = 0
    for q in queries:
        found, _ = rmi.search(q)
        hits += int(found)
    print("Sample accuracy:", hits, "/", queries.size)