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
  def __init__(self, image, mode):
    if mode == 1:
      print('This is mode', mode)
    elif mode == 2:
      print('This is mode', mode)
    elif mode == 3:
      print('This is mode', mode)
    elif mode == 4:
      print('This is mode', mode)
    elif mode == 5:
      DFT.test()
    else:
      print('Mode has to be between 1-5.')
      return