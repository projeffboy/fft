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

  def processImage(image):
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
      print('This is mode', mode)

      image, inputImage = self.processImage(image)

      compression = [0, 14, 30, 50, 70, 95]

      # write down abs of fft
      ft = DFT.fft_2d(image)
      
      figure, axis = plt.subplots(2, 3)

      for i in range(6):
        compression_level = compression[i]
        im_compressed = compress_image(ft, compression_level, inputImage.shape[0]*inputImage.shape[1])
        axis[i, j].imshow(np.real(im_compressed)[:inputImage.shape[0], :inputImage.shape[1]], plt.cm.gray)
        axis[i, j].set_title('{}% compression'.format(compression_level))

      figure.suptitle('Mode 3')
      plt.show()

      def compress_image(im_fft, compression_level, originalCount):
        if compression_level < 0 or compression_level > 100:
          AssertionError('compression_level must be between 0 to 100')

        rest = 100 - compression_level
        lower = np.percentile(im_fft, rest//2)
        upper = np.percentile(im_fft, 100 - rest//2)
        print('non zero values for level {}% are {} out of {}'.format(compression_level, int(originalCount * ((100 - compression_level) / 100.0)), originalCount))

        compressed_im_fft = im_fft * \
        np.logical_or(im_fft <= lower, im_fft >= upper)
        save_npz('coefficients-{}-compression.csr'.format(compression_level), csr_matrix(compressed_im_fft))

        return DFT.fast_two_dimension_inverse(compressed_im_fft)

    elif mode == 4:
      print('This is mode', mode)

      runs = 10

        # run plots
      fig, axis = plt.subplots()

      axis.set_xlabel('problem size')
      axis.set_ylabel('runtime in seconds')
      axis.set_title('Line plot with error bars')

      for algo_index, algo in enumerate([DFT.slow_two_dimension, DFT.fast_two_dimension]):
          x = []
          y = []

          problem_size = 2**4
          while problem_size <= 2**12:
            print("doing problem size of {}".format(problem_size))
            a = np.random.rand(int(math.sqrt(problem_size)),int(math.sqrt(problem_size)))
            x.append(problem_size)

            stats_data = []
            for i in range(runs):
              print("run {} ...".format(i+1))
              start_time = time.time()
              algo(a)
              delta = time.time() - start_time
              stats_data.append(delta)

              mean = statistics.mean(stats_data)
              sd = statistics.stdev(stats_data)

              print("for problem size of {} over {} runs: mean {}, stdev {}".format(problem_size, runs, mean, sd))

              y.append(mean)
              problem_size *= 4

          color = 'r--' if algo_index == 0 else 'g'
          plt.errorbar(x, y, yerr=sd, fmt=color)
      plt.show()

    elif mode == 5:
      DFT.test()
    else:
      print('Mode has to be between 1-5.')
      return