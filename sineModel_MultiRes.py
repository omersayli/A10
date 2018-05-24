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

def sineModel_MultiRes(x, fs, w1, w2, w3, N1, N2, N3, t, B1, B2, B3):
    """
    Week 10, Project:  A multi-resolution sinusoidal model
    Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
    Using Multi-resolution sine model
    x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB
    N1, N2, N3: size of the 3 complex spectrum
    B1, B2, B3: Frequency band edges (Upper limits) of the 3 frequency bands
    [0 - B1] is the first frequency band
    [B1 - B2] is the second frequency band
    [B2 - B3] is the third frequency band

    returns y: output array sound
    """


    hM1_1 = int(math.floor((w1.size + 1) / 2))  # half analysis for 1st window size by rounding
    hM1_2= int(math.floor(w1.size / 2))         # half analysis for 1st  window size by floor
    hM2_1 = int(math.floor((w2.size + 1) / 2))  # half analysis for 2nd window size by rounding
    hM2_2 = int(math.floor(w2.size / 2))        #half analysis for 2nd  window size by floor
    hM3_1 = int(math.floor((w3.size + 1) / 2))  # half analysis for 3rd window size by rounding
    hM3_2 = int(math.floor(w3.size / 2))        #half analysis for 3rd  window size by floor
    Ns = 512                                    # FFT size for synthesis (even)
    H = Ns // 4                                 # Hop size used for analysis and synthesis
    hNs = Ns // 2                               # half of synthesis FFT size
    pin = max(hNs, hM1_1, hM2_1, hM3_1)         # init sound pointer in middle of biggest anal window
    pend = x.size - pin                         # last sample to start a frame
    fftbuffer = np.zeros(max(N1, N2, N3))       # initialize buffer for FFT
    yw = np.zeros(Ns)                           # initialize output sound frame
    y = np.zeros(x.size)                        # initialize output array
    w_1 = w1 / sum(w1)                           # normalize the 1st analysis window
    w_2 = w2 / sum(w2)                           # normalize the 2nd analysis window
    w_3 = w3 / sum(w3)                           # normalize the 3rd analysis window
    sw = np.zeros(Ns)                           # initialize synthesis window
    ow = triang(2 * H)                          # triangular window
    sw[hNs - H:hNs + H] = ow                    # add triangular window
    bh = blackmanharris(Ns)                     # blackmanharris window
    bh = bh / sum(bh)                           # normalized blackmanharris window
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # normalized synthesis window

    while pin < pend:  # while input sound pointer is within sound
        # -----analysis-----
        #Selecting THREE FRAMES for the same CENTRAL PIN POINT, but FOR DIFFERENT LENGTHS
        x1 = x[pin - hM1_1:pin + hM1_2]  # select frame for window size 1
        x2 = x[pin - hM2_1:pin + hM2_2]  # select frame for window size2
        x3 = x[pin - hM3_1:pin + hM3_2]  # select frame for window size3
        mX1, pX1 = DFT.dftAnal(x1, w1, N1)  # compute dft for 1st frame
        mX2, pX2 = DFT.dftAnal(x2, w2, N2)  # compute dft for 2nd frame
        mX3, pX3 = DFT.dftAnal(x3, w3, N3)  # compute dft for 3rd frame
        ploc1 = UF.peakDetection(mX1, t)  # detect locations of peaks for 1st frame
        ploc2 = UF.peakDetection(mX2, t)  # detect locations of peaks for 2nd frame
        ploc3 = UF.peakDetection(mX3, t)  # detect locations of peaks for 3rd frame
        iploc1, ipmag1, ipphase1 = UF.peakInterp(mX1, pX1, ploc1)  # refine peak values by interpolation for the 1st frame
        iploc2, ipmag2, ipphase2 = UF.peakInterp(mX2, pX2, ploc2)  # refine peak values by interpolation for the 2nd frame
        iploc3, ipmag3, ipphase3 = UF.peakInterp(mX3, pX3, ploc3)  # refine peak values by interpolation for the 3rd frame
        ipfreq1 = fs * iploc1 / float(N1)  # convert peak locations of 1st frame to Hertz
        ipfreq2 = fs * iploc2 / float(N2)  # convert peak locations of 2nd frame to Hertz
        ipfreq3 = fs * iploc3 / float(N3)  # convert peak locations of 3rd frame to Hertz


        # Looking for indices of peak frequencies
        # in each band, for each window calculation
        indice_1 = np.logical_and(ipfreq1 > 0, ipfreq1 < B1)
        indice_2 = np.logical_and(ipfreq2 >= B1, ipfreq2 < B2)
        indice_3 = np.logical_and(ipfreq3 >= B2, ipfreq3 < B3)

        # Getting peaks which fall in selected frequency bands
        ipfreq = np.concatenate((ipfreq1[indice_1], ipfreq2[indice_2], ipfreq3[indice_3]))
        ipmag = np.concatenate((ipmag1[indice_1], ipmag2[indice_2], ipmag3[indice_3]))
        ipphase = np.concatenate((ipphase1[indice_1], ipphase2[indice_2], ipphase3[indice_3]))


        # -----synthesis-----
        Y = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)  # generate sines in the spectrum
        fftbuffer = np.real(ifft(Y))            # compute inverse FFT
        yw[:hNs - 1] = fftbuffer[hNs + 1:]      # undo zero-phase window
        yw[hNs - 1:] = fftbuffer[:hNs + 1]
        y[pin - hNs:pin + hNs] += sw * yw       # overlap-add and apply a synthesis window
        pin += H                                # advance sound pointer
    return y