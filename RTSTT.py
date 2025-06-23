
# import sounddevice as sd
# import numpy as np
# import queue
# from faster_whisper import WhisperModel

# q = queue.Queue()

# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     q.put(indata.copy())

# def main():
#     sample_rate = 16000
#     device_id = 12  

#     model = WhisperModel("distil-large-v3", device="cuda")

#     print(f"Using input device {device_id}... Press Ctrl+C to stop.")

#     buffer = np.zeros(0, dtype=np.float32)
#     chunk_duration = 5  # seconds
#     chunk_samples = chunk_duration * sample_rate

#     with sd.InputStream(samplerate=sample_rate, device=device_id, channels=1, dtype='float32', callback=audio_callback):
#         while True:
#             data = q.get()
#             buffer = np.concatenate((buffer, data.flatten()))
#             if len(buffer) >= chunk_samples:
#                 segments, _ = model.transcribe(buffer[:chunk_samples], beam_size=5)

#                 for segment in segments:
#                     print(segment.text)

#                 buffer = buffer[chunk_samples:]

# if __name__ == "__main__":
#     main()

#import torch
# import sounddevice as sd
# import numpy as np
# import queue
# from faster_whisper import WhisperModel
# import time

# q = queue.Queue()

# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(f"[STATUS] {status}")
#     q.put(indata.copy())

# def start_listening(sample_rate=16000, device_id=12, silence_timeout=1.5, chunk_duration=0.5):
#     print("[INFO] Loading Whisper model...")
#     model = WhisperModel("distil-large-v3", device="cuda")

#     print(f"[INFO] Starting input stream on device {device_id}")

#     vad_model, utils = torch.hub.load(
#         repo_or_dir='snakers4/silero-vad',
#         model='silero_vad',
#         force_reload=False,
#         onnx=False,
#         verbose=False
#     )
#     (get_speech_ts, _, _, _, _) = utils

#     buffer = np.zeros(0, dtype=np.float32)
#     speaking = False
#     silence_start = None

#     full_transcription = ""

#     with sd.InputStream(samplerate=sample_rate, device=device_id, channels=1, dtype='float32', callback=audio_callback):
#         print("[INFO] Listening for speech...")

#         while True:
#             data = q.get()
#             buffer = np.concatenate((buffer, data.flatten()))

#             if len(buffer) < int(chunk_duration * sample_rate):
#                 continue

#             audio_tensor = torch.from_numpy(buffer)
#             speech_timestamps = get_speech_ts(audio_tensor, vad_model, sampling_rate=sample_rate)

#             if len(speech_timestamps) > 0:
#                 if not speaking:
#                     print("[VOICE] Speech detected...")
#                 speaking = True
#                 silence_start = None

#                 # Transcribe current chunk
#                 segments, _ = model.transcribe(buffer, beam_size=1)
#                 partial_text = " ".join([seg.text for seg in segments]).strip()

#                 if partial_text:
#                     # Append partial transcription (avoid duplication)
#                     if not full_transcription.endswith(partial_text):
#                         full_transcription += (" " if full_transcription else "") + partial_text

#                 # Clear buffer for next chunk
#                 buffer = np.zeros(0, dtype=np.float32)

#                 print(f"[PARTIAL] {partial_text}")

#             else:
#                 if speaking:
#                     if silence_start is None:
#                         silence_start = time.time()
#                         print("[VOICE] Silence detected...")
#                     elif time.time() - silence_start > silence_timeout:
#                         # Yield full transcription when silence timeout reached
#                         if full_transcription.strip():
#                             print("[TRANSCRIBE] Finished speaking, yielding transcription...")
#                             yield full_transcription.strip()

#                         # Reset for next utterance
#                         full_transcription = ""
#                         buffer = np.zeros(0, dtype=np.float32)
#                         speaking = False
                        # silence_start = None


# import queue
# import sounddevice as sd
# import numpy as np
# import webrtcvad
# from faster_whisper import WhisperModel
# import collections
# import time

# q = queue.Queue()

# def audio_callback(indata, frames, time_info, status):
#     q.put(bytes(indata))


# def start_listening(sample_rate=16000, device_id=12, silence_timeout=1.0):
#     model = WhisperModel("distil-large-v3", device="cuda")
#     vad = webrtcvad.Vad(3)
#     voiced_frames = []
#     silence_blocks = 0
#     frame_duration = 30  # ms
#     bytes_per_frame = int(sample_rate * 2 * frame_duration / 1000)  # 2 bytes per sample
#     max_silence_blocks = int(silence_timeout * 1000 / frame_duration)
#     buffer = b""
#     frame_duration = 30  # ms
#     frame_size = int(sample_rate * frame_duration / 1000) * 2  # 2 bytes per sample (16-bit PCM)

#     while True:
#         buffer += q.get()

#         while len(buffer) >= frame_size:
#             frame = buffer[:frame_size]
#             buffer = buffer[frame_size:]

#             if vad.is_speech(frame, sample_rate):
#                 print("[VOICE] Speech detected...")
#                 voiced_frames.append(np.frombuffer(frame, dtype=np.int16))
#                 silence_blocks = 0
#             else:
#                 silence_blocks += 1
#                 if voiced_frames and silence_blocks > max_silence_blocks:
#                     audio = np.concatenate(voiced_frames, axis=0).flatten()
#                     segments, _ = model.transcribe(audio, beam_size=5)
#                     text = " ".join(segment.text for segment in segments)
#                     yield text
#                     voiced_frames.clear()
#                     silence_blocks = 0
# #WORKING VERSION

import sounddevice as sd
import numpy as np
import queue
import time
from faster_whisper import WhisperModel
from threading import Event

# Start with STT enabled


q = queue.Queue()
VOLUME_THRESHOLD_DB = -30  # dB
SILENCE_TIMEOUT = 1  # seconds
device_id= 12
sample_rate = 16000

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)

    volume_norm = np.linalg.norm(indata) / np.sqrt(len(indata))
    volume_db = 20 * np.log10(volume_norm + 1e-6)

    if volume_db > VOLUME_THRESHOLD_DB:
        q.put(indata.copy())

def is_loud_enough(data):
    volume_norm = np.linalg.norm(data) / np.sqrt(len(data))
    volume_db = 20 * np.log10(volume_norm + 1e-6)
    return volume_db > VOLUME_THRESHOLD_DB


def start_transcription(callback, device_id, sample_rate):
    model = WhisperModel("distil-large-v3", device="cuda")

    chunk_duration = 1  # seconds
    chunk_samples = chunk_duration * sample_rate

    buffer = np.zeros(0, dtype=np.float32)
    last_voice_time = time.time()
    last_audio_time = time.time()

    print(f"Using input device {device_id}... Press Ctrl+C to stop.")

    with sd.InputStream(samplerate=sample_rate, device=device_id, channels=1,
                        dtype='float32', callback=audio_callback):
        while True:

            try:
                data = q.get(timeout=0.1)
                if is_loud_enough(data):
                    buffer = np.concatenate((buffer, data.flatten()))
                    last_voice_time = time.time()
                elif time.time() - last_voice_time > SILENCE_TIMEOUT and len(buffer) > 0:
                    segments, _ = model.transcribe(buffer, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                    print(transcript)
                    callback(transcript)
                    buffer = np.zeros(0, dtype=np.float32)

            except queue.Empty:
                if buffer.size > 0 and len(buffer) >= chunk_samples:
                    segments, _ = model.transcribe(buffer, beam_size=5)
                    transcript = " ".join([segment.text for segment in segments])
                    print(transcript)
                    callback(transcript)
                    buffer = np.zeros(0, dtype=np.float32)


