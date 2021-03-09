from scipy.io import wavfile
from scipy import signal
from pydub import AudioSegment
import os
import numpy as np
import matplotlib.pyplot as plt

# Convert .m4a source file to .wav format, apply a highpass filter, and trim
def filterData(src_f, processed_f, freq_cutoff, start_time, end_time):
    sig = AudioSegment.from_file(src_f)
    start_ms = start_time*1000
    end_ms = end_time*1000
    trim_sig = sig[start_ms:end_ms]
    trim_sig.export("temp.wav", format="wav")
    sample_rate, sig = wavfile.read("temp.wav")

    # Apply a 10th degree, highpass butterworth filter
    filter = signal.butter(10, freq_cutoff, "hp", fs=sample_rate, output="sos")
    filtered_sig = signal.sosfilt(filter, sig)
    wavfile.write(processed_f, sample_rate,
                  filtered_sig.astype(np.int16))
    os.remove("temp.wav")

# Compute the spectrogram of the data and extract the peak frequencies
# at each time interval to obtain a peak frequencies plot
def getFigures(sound_f, spec_plt_name, freq_lower_lim, freq_upper_lim,
               freq_plt_name):
    sample_rate, sig = wavfile.read(sound_f)
    freqs, times, amplitudes = signal.spectrogram(sig, sample_rate)
    plt.pcolormesh(times, freqs, amplitudes, shading="gouraud")
    plt.colorbar()
    plt.title(spec_plt_name)
    plt.ylabel('Frequency [Hz]')
    plt.ylim(freq_lower_lim, freq_upper_lim)
    plt.xlabel('Time [sec]')
    #plt.show()

    peak_freq_indices = np.argmax(amplitudes, axis=0)
    peak_freqs = [freqs[i] for i in peak_freq_indices]
    plt.plot(times, peak_freqs)
    plt.title(freq_plt_name)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    #plt.show()
    return [times, peak_freqs]


# Get the avergage frequency over a time interval of a peak frquencies plot
def getAvgFreq(start_time, end_time, *data):
    start_ind = -1
    end_ind = -1
    times = data[0][0]
    for i in range(len(times)):
        if times[i] >= start_time and start_ind == -1:
            start_ind = i
        elif times[i] >= end_time and end_ind == -1:
            end_ind = i
            break

    peak_freqs = data[0][1]
    selected_freqs = peak_freqs[start_ind:end_ind+1]
    return sum(selected_freqs)/len(selected_freqs)

if __name__ == "__main__":

    #filterData("../audio_files/originals/7.2kHz.m4a",
               #"../audio_files/processed/7.2kHz.wav", 5000, 25, 27)
    data = getFigures("../audio_files/processed/7.2kHz.wav",
                           "Spectrogram (7.2kHz Trial)", 6700, 7800,
                           "STFT Peak Frequencies (7.2kHz Trial)")

    freq = getAvgFreq(1.16, 1.98, data)
    print(str(round(freq)) + "Hz")
