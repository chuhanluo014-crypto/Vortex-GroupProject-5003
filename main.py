from src.utils.data_loader import DatasetGenerator
from src.benchmarks.benchmark_runner import Benchmark

def main():
    print("🧠 Phase 1 – B-Tree Benchmark\n")
    size = 1000000
    print("#"*70)
    print(f"Testing {size:,} keys")
    print("#"*70)
    
    for name in ["Sequential (1,000,000)"]:
        keys = DatasetGenerator.generate_sequential(size)
        print(f"\n{'='*70}")
        print(f"Dataset: {name}  ({size:,} keys)")
        print(f"{'='*70}")
        Benchmark.run(name, keys)

if __name__ == "__main__":
    main()