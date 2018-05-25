"""
Week 10, Project:  A multi-resolution sinusoidal model
"""
import sys

import os

sys.path.append('../../software/models/')

import numpy as np
import math
from scipy.signal import blackmanharris, triang, get_window
from scipy.fftpack import ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF
import matplotlib.pyplot as plt

eps = np.finfo(float).eps



fs, x = UF.wavread('../../sounds/orchestra.wav')


w3 = get_window('blackman', 512)
w2 = get_window('blackman',2048)
w1 = get_window('blackman',4096)

N3 = 512
N2 = 2048
N1 = 4096

B1 = 1500
B2 = 5000
B3 = 22050

t = -100

y = sineModel_MultiRes(x, fs, w1, w2, w3, N1, N2, N3, t, B1, B2, B3)

rs = x - y
plt.plot(rs); plt.show()

UF.wavwrite(y, fs, 'output_multi_sin_orchestra_.wav')

y = sineModel(x, fs, w1, N1,t)
UF.wavwrite(y, fs, 'output_sin_orchestra.wav')