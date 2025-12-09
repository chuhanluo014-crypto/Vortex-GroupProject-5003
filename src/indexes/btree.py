"""
===============================================================================
B-TREE BASELINE IMPLEMENTATION
===============================================================================
This module defines a simple B-Tree used as the baseline for performance
comparison against learned index structures.

It currently supports:
    • Bulk build from sorted keys (read-only optimization)
    • Key search using binary search (logarithmic lookup)
    • Approximate memory usage reporting

The tree is built bottom-up from sorted arrays, creating leaf nodes and
then grouping them into internal nodes of configurable order (page size).

Usage:
    from src.indexes.btree import BTree
    from src.utils.data_loader import DatasetGenerator

    keys = DatasetGenerator.generate_uniform(1000)
    tree = BTree(order=64)
    tree.build_from_sorted_array(keys)
    print(tree.search(keys[50]))

Next Steps:
    - Implement dynamic insert/delete methods for update benchmarks.
    - Connect to learned indexes for hybrid experiments.
===============================================================================
"""

import bisect
import numpy as np


class BTreeNode:
    """Single node in a B-Tree."""

    def __init__(self, leaf: bool = True):
        self.keys = []        # Sorted list of keys
        self.children = []    # List of child BTreeNodes
        self.leaf = leaf      # True if leaf node


class BTree:
    """
    Simple B-Tree baseline implementation.
    Focused on search and bulk-load (no delete/insert by hand).
    """

    def __init__(self, order: int = 128):
        """
        Args:
            order: Maximum number of children per node (typical 64–256)
        """
        self.root = BTreeNode(leaf=True)
        self.order = order
        self.size = 0

    # ----------------------------------------------------------------------
    # Build
    # ----------------------------------------------------------------------
    def build_from_sorted_array(self, keys: np.ndarray):
        """Bulk-load a B-Tree from a sorted list of keys."""
        self.size = len(keys)
        if self.size == 0:
            return

        # Create leaf level
        leaf_nodes = []
        keys_per_leaf = self.order - 1
        for i in range(0, len(keys), keys_per_leaf):
            node = BTreeNode(leaf=True)
            node.keys = list(keys[i:i + keys_per_leaf])
            leaf_nodes.append(node)

        # Build internal levels
        current = leaf_nodes
        while len(current) > 1:
            next_level = []
            for i in range(0, len(current), self.order):
                parent = BTreeNode(leaf=False)
                group = current[i:i + self.order]
                for j, child in enumerate(group):
                    parent.children.append(child)
                    if j < len(group) - 1:
                        parent.keys.append(child.keys[0])
                next_level.append(parent)
            current = next_level

        self.root = current[0]

    # ----------------------------------------------------------------------
    # Search
    # ----------------------------------------------------------------------
    def search(self, key: float):
        """Search for a key; returns (found: bool, comparisons: int)."""
        return self._search_recursive(self.root, key)

    def _search_recursive(self, node: BTreeNode, key: float):
        idx = bisect.bisect_left(node.keys, key)
        if idx < len(node.keys) and node.keys[idx] == key:
            return True
        if node.leaf:
            return False
        return self._search_recursive(node.children[idx], key)

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------
    def get_memory_usage(self) -> int:
        """Approximate memory usage in bytes."""
        return self._mem_recursive(self.root)

    def _mem_recursive(self, node: BTreeNode) -> int:
        mem = len(node.keys) * 8 + len(node.children) * 8 + 16
        if not node.leaf:
            for child in node.children:
                mem += self._mem_recursive(child)
        return mem


    # ----------------------------------------------------------------------
    # Testing delete later if not needed.
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    from src.utils.data_loader import DatasetGenerator

    print("🔍 Testing B-Tree build and search...")

    keys = DatasetGenerator.generate_uniform(20)
    tree = BTree(order=4)
    tree.build_from_sorted_array(keys)

    found = tree.search(keys[5])
    missing = tree.search(9999999)

    print("Search existing key:", found)
    print("Search missing key:", missing)
    print("Estimated memory (bytes):", tree.get_memory_usage())

    print("✅ B-Tree test complete.")
