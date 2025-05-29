import pyaudio
import numpy as np
import wave
import threading

# ========== CONFIG ==========
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
FILTER_LEN = 512
# ============================

class NLMSFilter:
    def __init__(self, filter_len=512, mu=0.01, epsilon=1e-8):
        self.filter_len = filter_len
        self.mu = mu
        self.epsilon = epsilon
        self.w = np.zeros(filter_len)

    def process(self, mic_chunk, remote_chunk):
        N = len(mic_chunk)
        output = np.zeros(N)

        for n in range(self.filter_len, N):
            x = mic_chunk[n - self.filter_len:n]
            y = np.dot(self.w, x[::-1])
            e = remote_chunk[n] - y
            norm = np.dot(x, x) + self.epsilon
            self.w += (self.mu * e * x[::-1]) / norm
            output[n] = e
        return output.astype(np.int16)


def int16_to_np(data):
    return np.frombuffer(data, dtype=np.int16)


def np_to_int16(data):
    return data.astype(np.int16).tobytes()


pa = pyaudio.PyAudio()
mic_stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

remote_wave = wave.open("remote_mixed.wav", 'rb')
assert remote_wave.getframerate() == RATE
assert remote_wave.getnchannels() == CHANNELS
assert remote_wave.getsampwidth() == pa.get_sample_size(FORMAT)

output_stream = pa.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True,
                        frames_per_buffer=CHUNK)

nlms = NLMSFilter(filter_len=FILTER_LEN)

print("Processing... Press Ctrl+C to stop.")

try:
    while True:
        mic_data = mic_stream.read(CHUNK, exception_on_overflow=False)
        remote_data = remote_wave.readframes(CHUNK)

        if len(remote_data) < CHUNK * 2:
            break

        mic_np = int16_to_np(mic_data)
        remote_np = int16_to_np(remote_data)

        filtered_np = nlms.process(mic_np, remote_np)
        output_stream.write(np_to_int16(filtered_np))

except KeyboardInterrupt:
    print("\nStopped by user.")
finally:
    mic_stream.stop_stream()
    mic_stream.close()
    remote_wave.close()
    output_stream.stop_stream()
    output_stream.close()
    pa.terminate()

