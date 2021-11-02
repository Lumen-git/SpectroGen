import os
from librosa import *
import librosa.display
import IPython.display as ipd
import numpy as np
import matplotlib.pyplot as plt

input_file = "./Input"
output_file = "./Output"

def plot_spectrogram(inputf, Y, sr, hop_length, y_axis="linear"):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis="time",
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.savefig(str(output_file) + "/" + inputf)
    plt.close()

for filename in os.listdir(input_file):
    f = os.path.join(input_file,filename)
    if os.path.isfile(f):
        sound_file = str(f)
        ipd.Audio(sound_file)
        scale, sr = librosa.load(sound_file)
        FRAME_SIZE = 2048
        HOP_SIZE = 512
        S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        # print(S_scale.shape)
        # print(type(S_scale[0][0]))
        Y_scale = np.abs(S_scale) ** 2
        Y_log_scale = librosa.power_to_db(Y_scale)
        name = f.split("/")[-1].replace(".wav",".png")
        plot_spectrogram(name, Y_log_scale, sr, HOP_SIZE, y_axis="log")
