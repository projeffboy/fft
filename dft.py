import math

import numpy as np

class DFT:
  def naive_1d(x):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k, n in np.ndindex((N, N)):
      X[k] += x[n] * np.exp(-2j * np.pi / N * k * n)

    return X

  def naive_1d_inverse(X):
    X = np.array(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
      for k in range(N):
        x[n] += X[k] * np.exp(2j * np.pi / N * k * n)
      x[n] /= N

    return x

  def fft_1d(x):
    x = np.array(x, dtype=complex)
    N = len(x)

    if N % 2 != 0:
      raise Exception('Input must be a power of 2.')
    elif N <= 16:
      return DFT.naive_1d(x)
    else:
      X_even = DFT.fft_1d(x[::2])
      X_odd = DFT.fft_1d(x[1::2])
      X = np.zeros(N, dtype=complex)

      half_N = N // 2
      for k in range(N):
        X[k] = X_even[k % half_N] \
          + np.exp(-2j * np.pi / N * k) * X_odd[k % half_N]

      return X

  def fft_1d_inverse(X):
    X = np.array(X, dtype=complex)
    N = len(X)

    if N % 2 != 0:
      raise Exception('Input must be a power of 2.')
    elif N <= 16:
      return DFT.naive_1d_inverse(X)
    else:
      x_even = DFT.fft_1d_inverse(X[::2])
      x_odd = DFT.fft_1d_inverse(X[1::2])
      x = np.zeros(N, dtype=complex)

      half_N = N // 2
      for n in range(N):
        x[n] = x_even[n % half_N] \
          + np.exp(2j * np.pi / N * n) * x_odd[n % half_N]
        x[n] /= 2

      return x

  def naive_2d(f):
    f = np.array(f, dtype=complex)
    M, N = f.shape
    F = np.zeros((M, N), dtype=complex)

    for k, l, n, m in np.ndindex((M, N, N, M)):
      F[k, l] += f[m, n] * np.exp(-2j * np.pi * (k * m / M + l * n / N))

    return F

  # naive 2d inverse not necessary

  def fft_2d(f):
    return DFT._helper_2d(f, DFT.fft_1d)

  def fft_2d_inverse(F):
    return DFT._helper_2d(F, DFT.fft_1d_inverse, inverse=True)

  def _helper_2d(matrix, fn, inverse=False):
    matrix = np.array(matrix, dtype=complex)
    N, M = matrix.shape
    transform = np.zeros((N, M), dtype=complex)

    axis1 = 0
    axis2 = 1

    if inverse:
      axis1, axis2 = axis2, axis1

    transform = np.apply_along_axis(fn, axis1, matrix)
    transform = np.apply_along_axis(fn, axis2, transform)

    return transform

  def test(num_1d_samples=1024):
    if num_1d_samples % 2 != 0:
      raise Exception('Input must be a power of 2.')

    X_1d = np.random.random(num_1d_samples)
    fft = np.fft.fft(X_1d)

    num_2d_samples = int(math.sqrt(1024))
    X_2d = np.random.rand(num_2d_samples, num_2d_samples)
    fft2 = np.fft.fft2(X_2d)

    assertions = [
      (DFT.naive_1d, X_1d, fft),
      (DFT.naive_1d_inverse, fft, X_1d),
      (DFT.fft_1d, X_1d, fft),
      (DFT.fft_1d_inverse, fft, X_1d),
      (DFT.naive_2d, X_2d, fft2),
      (DFT.fft_2d, X_2d, fft2),
      (DFT.fft_2d_inverse, fft2, X_2d),
    ]

    for operator, discrete_fn, np_output in assertions:
      my_output = operator(discrete_fn)
      if np.allclose(my_output, np_output):
        print(operator.__name__ + ': PASSED')
      else:
        print(
f'''
{operator.__name__}: FAILED
Input:
{discrete_fn}
My Output Dimensions:
{my_output.shape}
My Output:
{my_output}
Expected Output Dimensions:
{np_output.shape}
Expected Output:
{np_output}
'''
        )