import librosa
import numpy as np
import matplotlib.pyplot as plt

# Path to your MP3 file
audio_file_path = 'mp3sample.mp3'

# Load audio data (Librosa automatically handles resampling and mono conversion)
audio, sample_rate = librosa.load(audio_file_path, sr=44100, mono=True)

# Apply FFT
fft_result = np.fft.fft(audio)
fft_magnitude = np.abs(fft_result)

# Calculate frequencies
freqs = np.fft.fftfreq(len(fft_result), 1/sample_rate)

# Only take the first half because the result is symmetrical
half_len = len(fft_result) // 2
freqs = freqs[:half_len]
fft_magnitude = fft_magnitude[:half_len]

# Find the fundamental frequency
fundamental_freq = freqs[np.argmax(fft_magnitude)]

# Print the fundamental frequency
print(f"The fundamental frequency of the audio segment: {fundamental_freq} Hz")

# Plotting (optional)
plt.plot(freqs, fft_magnitude)
plt.title("FFT Frequency Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, 2000)  # Limit viewing range from 0 to 2000 Hz
plt.show()
