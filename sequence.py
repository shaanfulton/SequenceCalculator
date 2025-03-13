import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
import sys

# This is to allow for larger numbers to be printed
sys.set_int_max_str_digits(0)

def sequence_calculator(n):
    """
    Calculate the nth number in the sequence S_n = 3*S_(n-1) - S_(n-2)
    with S_0 = 0 and S_1 = 1.
    """
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Matrix representation of the recurrence relation:
    # [S_n, S_(n-1)] = [[3, -1], [1, 0]] * [S_(n-1), S_(n-2)]
    #
    # We need to compute this matrix raised to the power (n-1).
    
    def matrix_multiply(A, B):
        C = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    C[i][j] += A[i][k] * B[k][j]
        return C
    
    def matrix_power(A, n):
        if n == 1:
            return A
        if n % 2 == 0:
            half_pow = matrix_power(A, n // 2)
            return matrix_multiply(half_pow, half_pow)
        else:
            half_pow = matrix_power(A, n // 2)
            return matrix_multiply(matrix_multiply(half_pow, half_pow), A)
    
    # Base matrix
    A = [[3, -1], [1, 0]]
    
    # Compute A^(n-1)
    result_matrix = matrix_power(A, n - 1)
    
    # [S_n, S_(n-1)] = A^(n-1) * [S_1, S_0] = A^(n-1) * [1, 0]
    return result_matrix[0][0] * 1 + result_matrix[0][1] * 0

def measure_execution_time(max_n=100000, step=1000):
    """Measure execution time for different values of n."""
    n_values = list(range(0, max_n + 1, step))
    if n_values[0] != 0:
        n_values = [0] + n_values
    if 1 not in n_values:
        n_values.append(1)
        n_values.sort()
    
    times = []
    
    for n in n_values:
        start_time = time.time()
        sequence_calculator(n)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return n_values, times

def log_function(x, a, b):
    """Function for curve fitting: a * log(x) + b."""
    # We use max(x, 1) to avoid log(0)
    return a * np.log(np.maximum(x, 1)) + b

def linear_function(x, a, b):
    """Function for curve fitting: a * x + b."""
    return a * x + b

def plot_execution_time():
    """Plot execution time vs n and perform curve fitting."""
    n_values, times = measure_execution_time()
    
    plt.figure(figsize=(10, 6))
    plt.scatter(n_values[2:], times[2:], label='Measured times')
    
    # Curve fitting, we try both logarithmic and linear models
    valid_indices = [i for i, n in enumerate(n_values) if n > 1]
    valid_n = np.array([n_values[i] for i in valid_indices])
    valid_times = np.array([times[i] for i in valid_indices])
    
    # Fit both models
    log_params, _ = curve_fit(log_function, valid_n, valid_times)
    linear_params, _ = curve_fit(linear_function, valid_n, valid_times)
    
    # Calculate R-squared for both models to determine better fit
    log_y_pred = log_function(valid_n, *log_params)
    linear_y_pred = linear_function(valid_n, *linear_params)
    
    log_residuals = valid_times - log_y_pred
    linear_residuals = valid_times - linear_y_pred
    
    ss_tot = np.sum((valid_times - np.mean(valid_times))**2)
    log_ss_res = np.sum(log_residuals**2)
    linear_ss_res = np.sum(linear_residuals**2)
    
    log_r_squared = 1 - (log_ss_res / ss_tot)
    linear_r_squared = 1 - (linear_ss_res / ss_tot)
    
    # Generate points for the fitted curves
    x_fit = np.linspace(min(valid_n), max(valid_n), 100)
    log_y_fit = log_function(x_fit, *log_params)
    linear_y_fit = linear_function(x_fit, *linear_params)
    
    # Plot both models
    plt.plot(x_fit, log_y_fit, 'r-', label=f'Log model: {log_params[0]:.6f}*log(n)+{log_params[1]:.6f} (R²={log_r_squared:.4f})')
    plt.plot(x_fit, linear_y_fit, 'g-', label=f'Linear model: {linear_params[0]:.6f}*n+{linear_params[1]:.6f} (R²={linear_r_squared:.4f})')
    
    plt.xlabel('n')
    plt.ylabel('Execution time (seconds)')
    plt.title('Execution Time vs n')
    plt.legend()
    plt.grid(True)
    plt.savefig('execution_time_plot.png')
    plt.show()

if __name__ == "__main__":
    # Calculate and print the 100,000th number in the sequence
    result = sequence_calculator(100000)
    print(f"S_100000 = {result}")
    
    # Plot execution time
    plot_execution_time()