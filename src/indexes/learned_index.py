"""
===============================================================================
LINEAR MODEL INDEX (LEARNED INDEX BASELINE)
===============================================================================
This module implements a simple learned index using a single linear regression
model. It predicts the position of each key in a sorted array, then performs
a local binary search around that prediction for correction.

Usage:
    from src.indexes.linear_index import LearnedIndex
    from src.utils.data_loader import DatasetGenerator

    keys = DatasetGenerator.generate_uniform(10000)
    index = LearnedIndex()
    index.build_from_sorted_array(keys)
    print(index.search(keys[100]))
===============================================================================
"""

import numpy as np
import bisect


class LearnedIndex:
    """Simple learned index using linear regression and local search correction."""

    def __init__(self):
        self.a = 0.0  # slope
        self.b = 0.0  # intercept
        self.window = 64  # default search window size
        self.keys = None  # the sorted keys
        self.correct_predictions = 0  # tracking correct predictions
        self.fallbacks = 0  # tracking fallbacks to full search
        self.false_negatives = 0  # tracking false negatives / wrong predictions
        self.not_found = 0  # tracking not found cases
        self.total_queries = 0  # total queries made

    # ----------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------
    def build_from_sorted_array(self, keys: np.ndarray):
        """Fit a linear regression model to predict key position."""
        self.keys = keys
        n = len(keys)
        if n == 0:
            print("Warning: Building LearnedIndex with empty key array.")
            return

        # Normalize positions (0 to n-1)
        positions = np.arange(n)

        # Fit simple linear regression: position ≈ a * key + b
        self.a, self.b = np.polyfit(keys, positions, 1)

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(self, key: float) -> bool:
        """Predict approximate position, then correct locally.
        Args:
            key: The key to search for.
        Returns:
            bool indicating whether the key was found.
        """
        if self.keys is None or len(self.keys) == 0:
            return False

        n = len(self.keys)
        self.total_queries += 1

        # Predict approximate index
        pred = int(self.a * key + self.b)

        # Clamp to valid range
        pred = max(0, min(n - 1, pred))

        # Define local search window
        left = max(0, pred - self.window)
        right = min(n, pred + self.window)

        # Local binary search correction
        idx = bisect.bisect_left(self.keys[left:right], key)

        # try this TODO
        #idx = np.searchsorted(self.keys[left:right], key, side='left')

        found = (idx + left < n) and (self.keys[idx + left] == key)

        #for debug tracking
        if found:
            self.correct_predictions += 1 # predicted position was correct
        elif not found:
            # fall back to full search
            self.fallbacks += 1
            # Perform full search as fallback
            full_idx = bisect.bisect_left(self.keys, key)
            if full_idx < n and self.keys[full_idx] == key:
                found = True
                self.false_negatives += 1
            else:
                self.not_found += 1
        return found

    # ----------------------------------------------------------------------
    # Memory usage estimate
    # ----------------------------------------------------------------------
    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes."""
        return len(self.keys) * 8 + 16 + 16  # keys + a/b params + overhead
        
    # ----------------------------------------------------------------------
    # Prediction accuracy
    # ----------------------------------------------------------------------
    def predict(self, key: float) -> int:

        pred = int(self.a * key + self.b)
        n = len(self.keys) if self.keys is not None else 0
        # Clamp to valid range
        pred = max(0, min(n - 1, pred))
        return pred
    

# ----------------------------------------------------------------------
# Quick sanity check / debug test
# use in console to run : python -m src.indexes.learned_index  
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import numpy as np
    import time
    from ..utils.data_loader import DatasetGenerator

    print("Sanity Check: LearnedIndex\n")

    # Test different dataset types
    for name, gen_func in [
        ("Sequential", DatasetGenerator.generate_sequential),
        ("Uniform", DatasetGenerator.generate_uniform),
        ("Mixed", DatasetGenerator.generate_mixed),
    ]:
        print(f"\n{'='*60}")
        print(f"Dataset: {name}")
        print(f"{'='*60}")

        # Generate and build
        keys = gen_func(100_000)
        index = LearnedIndex()
        index.build_from_sorted_array(keys)

                        # ---- Mini-benchmark (same methodology as Benchmark.run) ----

        num_queries = 1000
        # Build timing (rebuild once just for timing fairness)
        t0 = time.perf_counter()
        index2 = LearnedIndex()
        index2.build_from_sorted_array(keys)
        t1 = time.perf_counter()
        build_ms = (t1 - t0) * 1000.0

        # Query set: half existing, half random in-range
        rng = np.random.default_rng(0)  # seed for reproducibility
        existing = rng.choice(keys, num_queries // 2, replace=True)
        randoms  = rng.uniform(keys.min(), keys.max(), num_queries // 2)
        queries  = np.concatenate([existing, randoms])
        rng.shuffle(queries)

        # Helpers to accept either bool or (bool, idx)
        def _found_only(res):
            if isinstance(res, tuple):
                return bool(res[0])
            return bool(res)

        # Lookup timing + hit/miss breakdown
        times_ns = []
        hits = misses = 0
        hits_existing = misses_existing = 0
        hits_randoms = misses_randoms = 0

        # First half of queries correspond to existing before shuffle; since we shuffled,
        # we’ll compute hit/miss categories by testing membership.
        # (This avoids carrying labels through the shuffle.)
        keyset = set(keys.tolist())  # membership test for label (ok for this small mini-bench)

        for q in queries:
            t0 = time.perf_counter()
            res = index2.search(q)
            t1 = time.perf_counter()
            times_ns.append((t1 - t0) * 1e9)

            found = _found_only(res)
            if found:
                hits += 1
            else:
                misses += 1

            # categorize as "existing" vs "random" by set membership
            if q in keyset:
                if found: hits_existing += 1
                else:     misses_existing += 1
            else:
                if found: hits_randoms += 1
                else:     misses_randoms += 1

        avg_ns = float(np.mean(times_ns))
        mem_mb = index2.get_memory_usage() / (1024 * 1024)

        print("\n-- Mini-Benchmark (LearnedIndex) --")
        print(f"Build: {build_ms:8.2f} ms | Lookup mean: {avg_ns:8.2f} ns | Mem: {mem_mb:6.3f} MB")
        print(f"Hits: {hits}  Misses: {misses}  |  Hits(existing/random): {hits_existing}/{hits_randoms}  "
              f"Misses(existing/random): {misses_existing}/{misses_randoms} \n\n")
        # -------------------------------------------------------------

        # Print model parameters and error window
        print(f"Slope (a): {index.a:.6f}")
        print(f"Intercept (b): {index.b:.6f}")
        print(f"Memory usage: {index.get_memory_usage() / 1024:.2f} KB")

        # Test a few existing keys
        test_keys = [
            keys[0],                      # first
            keys[len(keys)//2],           # middle
            keys[-1],                     # last
        ]

        for k in test_keys:
            found = index.search(k)
            print(f"Key={k:.2f} -> Found={found}")

        # Test a random non-existing key
        q = float(np.random.uniform(keys.min(), keys.max()))
        found = index.search(q)
        print(f"\nRandom query: {q:.2f} -> Found={found}")
    
    print("\nLearnedIndex sanity check complete.\n")
   
