# -----------------------
#
# Copyright Jin Fagang @2018
# 
# 12/29/18
# read_audio
# -----------------------
import librosa
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def load_wav(p, sr):
    """
    sr: sample rate
    :param p:
    :param sr:
    :return:
    """
    return librosa.core.load(p, sr=sr)[0]

def read_audio(f):
    wav = load_wav(f, 22050)
    print(wav)
    print(wav.shape)
    # plt.plot([i for i in range(wav.shape[0])], wav)
    # plt.show()


if __name__ == '__main__':
    read_audio('LJ001-0001.wav')
