## Sequence Calculator Overview

An efficient implementation to calculate the nth number in a recurrence relation.

### Problem Statement
Calculate the nth number of the sequence S_n, defined as:
- S_n = 3*S_(n-1) - S_(n-2)
- With initial conditions: S_0 = 0 and S_1 = 1

### Implementation Details
- Uses matrix exponentiation to achieve O(log n) theoretical time complexity
- Includes performance measurement and visualization
- Handles very large values of n (up to 100,000+)

### Time Complexity Analysis

#### Theoretical Complexity
The implementation uses matrix exponentiation with the divide-and-conquer approach:
1. The recurrence relation is represented as a 2×2 matrix multiplication
2. Matrix exponentiation is performed in O(log n) time using the divide-and-conquer approach

#### Empirical Results
While the theoretical time complexity is O(log n), our empirical measurements show that the execution time follows a more linear pattern. After fitting both logarithmic and linear models to the data, the linear model provides a better fit with a higher R² value.

#### Explaining the Discrepancy
Several factors contribute to this discrepancy between theoretical and observed complexity:

1. **Hidden Linear Factor**: The matrix exponentiation algorithm has O(log n) complexity in terms of the number of matrix multiplications, but each multiplication involves integers whose size grows linearly with n. This creates an additional O(n) factor in the actual runtime.

2. **Memory Management Overhead**: As the integers grow larger, Python's memory management system spends more time allocating and deallocating memory, adding overhead that scales with n.

#### Actual Complexity
The actual time complexity can be more accurately described as O(n log n) when accounting for the cost of arbitrary-precision integer operations:
- O(log n) matrix multiplications
- Each multiplication costs O(n) due to the growing size of the integers
- Therefore, the overall complexity is O(n log n)

For practical purposes within the tested range (n ≤ 100,000), the empirical results suggest that the algorithm behaves more like O(n), as the linear component dominates the observed performance.

## Dependencies
This implementation requires NumPy for numerical computations and matrix operations, Matplotlib for plotting and visualization, and SciPy for curve fitting analysis.

### How to Run
```bash
pip install -r requirements.txt
python sequence.py
```