#!/usr/bin/python3.9

import time
import math
import statistics

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.sparse import csr_matrix, save_npz

from dft import DFT

class Modes:

  def processImage(self, image):
      inputImage = plt.imread(image).astype(float)

      x = int(pow(2, int(math.log(inputImage.shape[0], 2))+1))
      y = int(pow(2, int(math.log(inputImage.shape[1], 2))+1))
      size = (x,y)
      image = np.zeros(size)
      image[:inputImage.shape[0], :inputImage.shape[1]] = inputImage

      return image, inputImage


  def __init__(self, image, mode):
  
    if mode == 1:
      print('This is mode ', mode)

      image, inputImage = self.processImage(image)

      ft = DFT.fft_2d(image)
      pltX = image[:inputImage.shape[0]]
      pltY = image[:inputImage.shape[1]]

      figure, axis = plt.subplots(1, 2)
      axis[0].imshow(pltX, pltY, plt.cm.gray)
      axis[1].imshow(np.abs(ft), norm=colors.Normalize())
      axis[1].set_title('Fourier Transform 2D')
      figure.suptitle('Mode 1')
      plt.show()

    elif mode == 2:
      print('This is mode', mode)
      
      image, inputImage = self.processImage(image)

      ft = DFT.fft_2d(image, withRatio=True, ratio=0.15)
      ift = DFT.fft_2d_inverse(ft)

      pltX = image[:inputImage.shape[0]]
      pltY = image[:inputImage.shape[1]]

      figure, axis = plt.subplots(1, 2)
      axis[0].imshow(pltX, pltY, plt.cm.gray)
      axis[1].imshow(ift[:inputImage.shape[0], :inputImage.shape[1]], plt.cm.gray)
      axis[1].set_title('Fourier Transform 2D w/o Noise')
      figure.suptitle('Mode 2')

      plt.show()

    elif mode == 3:
      def compress_image(img_fft, compress_percent, ogCount):
        marker = compress_percent // 2
        lower = np.percentile(img_fft, 50 - marker)
        upper = np.percentile(img_fft, 50 + marker)
        print('{}% compressed: {}/{} non-zero values'.format(
          compress_percent,
          int(ogCount * (100 - compress_percent) / 100),
          ogCount
        ))

        compressed_img_fft = img_fft \
          * np.logical_or(img_fft <= lower, img_fft >= upper)
        save_npz(
          'compression-matrices/{}-compression.csr'.format(compress_percent),
          csr_matrix(compressed_img_fft)
        )

        return DFT.fft_2d_inverse(compressed_img_fft)

      print('This is mode', mode)

      image, inputImage = self.processImage(image)

      # modify this to change the compression levels
      compression = [0, 19, 19 * 2, 19 * 3, 19 * 4, 19 * 5]

      ft = DFT.fft_2d(image)
      
      figure, axis = plt.subplots(2, 3)

      for i, j in np.ndindex((2, 3)):
        compression_level = compression[j + i * 3]
        img_compressed = compress_image(
          ft, compression_level, inputImage.shape[0] * inputImage.shape[1]
        )
        axis[i, j].imshow(
          np.real(img_compressed)[:inputImage.shape[0],
          :inputImage.shape[1]],
          plt.cm.gray
        )
        axis[i, j].set_title('{}% Compressed'.format(compression_level))

      plt.show()

    elif mode == 4:
      print('This is mode', mode)

      runs = 10 # important parameter

        # run plots
      _, axis = plt.subplots()

      axis.set_xlabel('Problem Size')
      axis.set_ylabel('Algorithm Runtime (s)')
      axis.set_title('Naive vs FFT Runtime Comparison')

      for algo_i, algorithm in enumerate([DFT.naive_2d_efficient, DFT.fft_2d]):
        x = []
        y = []
        stdev_arr = []
        problem_size = 2 ** 5 # the smallest 2d array

        while problem_size <= 2 ** 8: # the largest 2d array, 9 and 10 take too long
          print('Problem size:', problem_size)
          sq = np.random.rand(problem_size, problem_size)
          x.append(problem_size)

          stats_data = []
          for i in range(runs):
            print('Run', i + 1, '...')
            start_time = time.time()
            algorithm(sq)
            delta = time.time() - start_time
            stats_data.append(delta)

          mean = statistics.mean(stats_data)
          stdev = statistics.stdev(stats_data)

          print(
f'''For problem size {problem_size} over {runs} runs:
mean: {mean}
stdev: {stdev}
'''
          )

          y.append(mean)
          stdev_arr.append(stdev)
          problem_size *= 2

        color = 'r' if algo_i == 0 else 'k'
        plt.errorbar(x, y, yerr=stdev_arr, fmt=color)
      plt.show()

    elif mode == 5:
      DFT.test()
    else:
      print('Mode has to be between 1-5.')
      return