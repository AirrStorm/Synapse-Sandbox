import numpy as np

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
            x = remote_chunk[n - self.filter_len:n]  # Echo signal
            y = np.dot(self.w, x[::-1])
            e = mic_chunk[n] - y  # Actual mic signal - estimated echo
            norm = np.dot(x, x) + self.epsilon
            self.w += (self.mu * e * x[::-1]) / norm
            output[n] = e

        return output.astype(np.float32)


# Utility conversion helpers

def int16_to_np(data):
    return np.frombuffer(data, dtype=np.int16)

def np_to_int16(data):
    return data.astype(np.int16).tobytes()
