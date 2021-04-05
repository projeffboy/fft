import math

import numpy as np

class DFT:
  def naive_1d(x):
    return DFT._naive_1d_helper(x)

  def naive_1d_inverse(X):
    return DFT._naive_1d_helper(X, inverse=True)

  def _naive_1d_helper(x, inverse=False):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    mult = 1 if inverse else -1

    for k in range(N):
      for n in range(N):
        X[k] += x[n] * np.exp(mult * 2j * np.pi / N * k * n)
      if inverse:
        X[k] /= N

    return X

  def fft_1d(x):
    return DFT._fft_1d_helper(x, DFT.naive_1d)

  def fft_1d_inverse(X):
    return DFT._fft_1d_helper(X, DFT.naive_1d_inverse, inverse=True)

  def _fft_1d_helper(x, base_fn, inverse=False):
    x = np.array(x, dtype=complex)
    N = len(x)

    if N % 2 != 0:
      raise Exception('Input must be a power of 2.')
    elif N <= 16:
      return base_fn(x)
    else:
      X_even = DFT._fft_1d_helper(x[::2], base_fn, inverse)
      X_odd = DFT._fft_1d_helper(x[1::2], base_fn, inverse)
      X = np.zeros(N, dtype=complex)

      half_N = N // 2
      mult = 1 if inverse else -1
      for k in range(N):
        X[k] = X_even[k % half_N] \
          + np.exp(mult * 2j * np.pi / N * k) * X_odd[k % half_N]
        if inverse:
          X[k] /= 2

      return X

  def naive_2d(f):
    f = np.array(f, dtype=complex)
    M, N = f.shape
    F = np.zeros((M, N), dtype=complex)

    for k, l, n, m in np.ndindex((M, N, N, M)):
      F[k, l] += f[m, n] * np.exp(-2j * np.pi * (k * m / M + l * n / N))

    return F

  # not fft but more efficient version of naive
  def naive_2d_efficient(f):
    return DFT._helper_2d(f, DFT.naive_1d)

  # naive 2d inverse not necessary

  def fft_2d(f):
    return DFT._helper_2d(f, DFT.fft_1d)

  def fft_2d_inverse(F):
    return DFT._helper_2d(F, DFT.fft_1d_inverse, inverse=True)

  def _helper_2d(matrix, fn, inverse=False, withRatio=False, ratio=0.00):
    matrix = np.array(matrix, dtype=complex)
    N, M = matrix.shape
    transform = np.zeros((N, M), dtype=complex)

    axis1 = 0
    axis2 = 1

    if inverse:
      axis1, axis2 = axis2, axis1

    transform = np.apply_along_axis(fn, axis1, matrix)
    transform = np.apply_along_axis(fn, axis2, transform)

    if withRatio:
      inv_ratio = (1-ratio)
      rows, columns = transform.shape
      transform[int(rows*ratio):int(rows*(inv_ratio))] = 0
      transform[:, int(columns*ratio):int(columns*(inv_ratio))] = 0

    return transform

  def test(num_1d_samples=1024):
    if num_1d_samples % 2 != 0:
      raise Exception('Input must be a power of 2.')

    X_1d = np.random.random(num_1d_samples)
    fft_1d = np.fft.fft(X_1d)

    num_2d_samples = int(math.sqrt(1024))
    X_2d = np.random.rand(num_2d_samples, num_2d_samples)
    fft_2d = np.fft.fft2(X_2d)

    assertions = [
      (DFT.naive_1d, X_1d, fft_1d),
      (DFT.naive_1d_inverse, fft_1d, X_1d),
      (DFT.fft_1d, X_1d, fft_1d),
      (DFT.fft_1d_inverse, fft_1d, X_1d),
      (DFT.naive_2d, X_2d, fft_2d),
      (DFT.naive_2d_efficient, X_2d, fft_2d),
      (DFT.fft_2d, X_2d, fft_2d),
      (DFT.fft_2d_inverse, fft_2d, X_2d),
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