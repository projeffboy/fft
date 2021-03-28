#!/usr/bin/python3.9

import argparse

from modes import Modes

def __main__():
  parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
  parser.add_argument(
    '-m',
    dest='mode',
    help='''
- [1] (Default) for fast mode where the image is converted into its FFT form and displayed
- [2] for denoising where the image is denoised by applying an FFT, truncating high 
frequencies and then displayed
- [3] for compressing and saving the image
- [4] for plotting the runtime graphs for the report
- [5] (Extra mode) for testing if our output is the same as numpy's fft methods
    ''',
    type=int,
    default=1
  )
  parser.add_argument(
    '-i',
    dest='image',
    help='filename of the image we wish to take the DFT of (default: moonlanding.png)',
    type=str,
    default='moonlanding.png'
  )
  arguments = parser.parse_args()
  Modes(arguments.image, arguments.mode)

if __name__ == '__main__':
  __main__()
