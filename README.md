# Recursive Model Index (RMI) Replication 

Replication of the **Recursive Model Index (RMI)** from the paper **"The Case for Learned Index Structures"** (SIGMOD 2018). This project implements a prototype of the RMI and an optimized B-Tree baseline in Python to benchmark their performance.

##  Project Overview

The project aims to validate the concept of "index as model," replacing traditional indexes with machine learning models to map a key $x$ to its position $P$ in an array.

* **RMI Core:** A **Two-Stage** hierarchical index structure that learns the data's Cumulative Distribution Function (CDF).
* **Lookup:** RMI provides a predicted position $P_{predicted}$ and then performs a fast **Binary Search** within a minimal **Error Window ($\approx 2\epsilon$)** to find the precise location.
* **B-Tree Baseline:** An optimized, bulk-built B-Tree used as a standard $O(\log N)$ comparison baseline.

---

##  Dataset Information

Benchmarks were conducted on synthetic datasets ($N=10,000,000$ records) generated to test various data distributions:

* **Sequential**: Ordered numbers (0, 1, 2, ...); **Best-case scenario** for model fitting.
* **Uniform**: Keys are randomly and evenly spread across a range.
* **Mixed**: Combination of uniform distribution and clustered groups to simulate **real-world data skew**; poses the highest challenge

---

##  Key Replication Findings (at $N=10^7$)

The experiment highlights the gap between RMI's theoretical complexity and its practical performance in a high-level language environment.

### 1. Lookup Latency
B-Tree was **significantly faster** than RMI across all data distributions.

* **Conclusion:** The massive **constant time overhead** introduced by the Python interpreter during model prediction and function calls far outweighed the RMI's theoretical $O(\log \epsilon)$ advantage, leading to slower absolute query speeds.

### 2. Memory Usage
RMI consistently showed **lower memory usage** than B-Tree, confirming its efficiency principle.

* **Observation:** The memory saving was approximately $1.6\%$ to $5\%$—much less than the theoretical order-of-magnitude savings.
* **Reason:** The memory footprint was dominated by the large **metadata and object overhead** of Python structures (like class instances and NumPy arrays), which obscured the savings from the minimal model parameters.

---
