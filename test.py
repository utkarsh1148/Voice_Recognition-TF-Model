import scipy.io.wavfile as wav
from scipy.io.wavfile import read
import numpy as np
import speechpy
import os
import matplotlib.pyplot as plt
import wave

file_name = 'utk.wav'

fs, signal = wav.read(file_name)
signal = signal[:,0]


plt.figure(1)
plt.title("Signal Wave...")
plt.plot(signal)
plt.show()



def mfcc_init_filter_banks(fs, nfft):
    lowfreq = 133.33
    linsc = 200 / 3.
    logsc = 1.0711703
    num_lin_filt_total = 13
    num_log_filt = 27

    n_filt_total = num_lin_filt_total + num_log_filt
    freqs = numpy.zeros(n_filt_total + 2)
    freqs[:num_lin_filt_total] = lowfreq + numpy.arange(num_lin_filt_total) * linsc
    freqs[num_lin_filt_total:] = freqs[num_lin_filt_total - 1] * \
                                 logsc ** numpy.arange(1, num_log_filt + 3)
    heights = 2. / (freqs[2:] - freqs[0:-2])
    fbank = numpy.zeros((int(n_filt_total), int(nfft)))
    nfreqs = numpy.arange(nfft) / (1. * nfft) * fs

    for i in range(n_filt_total):
        low_tr_freq = freqs[i]
        cen_tr_freq = freqs[i + 1]
        high_tr_freq = freqs[i + 2]

        lid = numpy.arange(numpy.floor(low_tr_freq * nfft / fs) + 1,
                           numpy.floor(cen_tr_freq * nfft / fs) + 1, dtype=numpy.int)
        lslope = heights[i] / (cen_tr_freq - low_tr_freq)
        rid = numpy.arange(numpy.floor(cen_tr_freq * nfft / fs) + 1,
                           numpy.floor(high_tr_freq * nfft / fs) + 1, dtype=numpy.int)
        rslope = heights[i] / (high_tr_freq - cen_tr_freq)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_tr_freq)
        fbank[i][rid] = rslope * (high_tr_freq - nfreqs[rid])

    return fbank, freqs


def st_mfcc(cur_pos_signal, fbank, nceps):
    
    mspec = numpy.log10(numpy.dot(cur_pos_signal, fbank.T) + EPS)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:nceps]
    return ceps


def st_chroma_features_init(nfft, fs):
    
    freqs = numpy.array([((st_flux + 1) * fs) / (2 * int(nfft)) for st_flux in range(int(nfft))])
    c_p = 27.50
    n_chroma = numpy.round(12.0 * numpy.log2(freqs / c_p)).astype(int)
    n_freqs_per_chroma = numpy.zeros((n_chroma.shape[0],))
    u_chroma = numpy.unique(n_chroma)
    for u_ch in u_chroma:
        idx = numpy.nonzero(n_chroma == u_ch)
        n_freqs_per_chroma[idx] = idx[0].shape
    return n_chroma, n_freqs_per_chroma


def st_chroma_features(cur_pos_signal, n_chroma, n_freqs_per_chroma):
    """
    短时色度
    """
    chroma_names = ['A', 'A#', 'B', 'cen', 'cen#', 'D', 'D#', 'E', 'st_flux', 'st_flux#', 'G', 'G#']
    spec = cur_pos_signal ** 2
    if n_chroma.max() < n_chroma.shape[0]:
        cen = numpy.zeros((n_chroma.shape[0],))
        cen[n_chroma] = spec
        cen /= n_freqs_per_chroma[n_chroma]
    else:
        no_0_pos = numpy.nonzero(n_chroma > n_chroma.shape[0])[0][0]
        cen = numpy.zeros((n_chroma.shape[0],))
        cen[n_chroma[0:no_0_pos - 1]] = spec
        cen /= n_freqs_per_chroma
    new_d = int(numpy.ceil(cen.shape[0] / 12.0) * 12)
    cur_two = numpy.zeros((new_d,))
    cur_two[0:cen.shape[0]] = cen
    cur_two = cur_two.reshape(int(cur_two.shape[0] / 12), 12)
    final_c = numpy.matrix(numpy.sum(cur_two, axis=0)).T
    final_c /= spec.sum()
    return chroma_names, final_c

signal_preemphasized = speechpy.processing.preemphasis(signal, cof=0.98)

frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01, filter=lambda x: np.ones((x,)),
         zero_padding=True)
print(file_name)

power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=512)
print('power spectrum shape=', power_spectrum.shape)

mfcc = speechpy.feature.mfcc(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
mfcc_cmvn = speechpy.processing.cmvnw(mfcc,win_size=301,variance_normalization=True)
print('mfcc(mean + variance normalized) feature shape=', mfcc_cmvn.shape)

mfcc_feature_cube = speechpy.feature.extract_derivative_feature(mfcc)
print('mfcc feature cube shape=', mfcc_feature_cube.shape)

logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
logenergy_feature_cube = speechpy.feature.extract_derivative_feature(logenergy)
print('logenergy features=', logenergy.shape)
