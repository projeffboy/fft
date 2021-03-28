import math

import numpy as np

class DFT:
  @staticmethod
  def naive_1d(x):
    x = np.array(x, dtype=complex)
    N = len(x)
    X = np.zeros(N, dtype=complex)

    for k, n in np.ndindex((N, N)):
      X[k] += x[n] * np.exp(-2j * np.pi / N * k * n)

    return X

  @staticmethod
  def naive_1d_inverse(X):
    X = np.array(X, dtype=complex)
    N = len(X)
    x = np.zeros(N, dtype=complex)

    for n in range(N):
      for k in range(N):
        x[n] += X[k] * np.exp(2j * np.pi / N * k * n)
      x[n] /= N

    return x

  @staticmethod
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

  @staticmethod
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

  @staticmethod
  def naive_2d(x):
    x = np.array(x, dtype=complex)
    N, M = x.shape
    X = np.zeros((N, M), dtype=complex)

    for k, l, m, n in np.ndindex((N, M, M, N)):
      X[k, l] += x[n, m] * np.exp(-2j * np.pi * (l * m / M + k * n / N))

    return X

  # naive 2d inverse not necessary

  @staticmethod
  def fft_2d(x):
    x = np.array(x, dtype=complex)
    N, M = x.shape
    X = np.zeros((N, M), dtype=complex)

    for col in range(M):
      X[:, col] = DFT.fft_1d(x[:, col])

    for row in range(N):
      X[row, :] = DFT.fft_1d(X[row, :])

    return X

  @staticmethod
  def fft_2d_inverse(F):
    F = np.array(F, dtype=complex)
    N, M = F.shape
    f = np.zeros((N, M), dtype=complex)

    for row in range(N):
      f[row, :] = DFT.fft_1d_inverse(F[row, :])

    for col in range(M):
      f[:, col] = DFT.fft_1d_inverse(f[:, col])

    return f

  @staticmethod
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

    for operator, discrete_fn, np_answer in assertions:
      if np.allclose(operator(discrete_fn), np_answer):
        print(operator.__name__ + ': PASSED')
      else:
        print(
f'''
{operator.__name__}: FAILED
Input:\n{discrete_fn}
Output:\n{operator(discrete_fn)}
Expected Output:\n{np_answer}
'''
        )
