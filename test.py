# from RealtimeSTT import AudioToTextRecorder

# def process_text(text):
#     print(text)

# if __name__ == '__main__':
#     print("Wait until it says 'speak now'")
#     recorder = AudioToTextRecorder(input_device_index=12)

#     while True:
#         recorder.text(process_text)


## PYAUDIO 
# import pyaudio
# import numpy as np
# from whisper import load_model
# import time
# import re
# import threading
# import queue

# # Config
# CHUNK = 1024
# RATE = 16000
# CHANNELS = 1
# THRESHOLD = 0.02
# BUFFER_DURATION = 2
# DEBOUNCE_INTERVAL = 3

# # Model
# model = load_model("base", device="cuda")

# # Queue for audio buffers
# audio_queue = queue.Queue()

# # PyAudio setup
# p = pyaudio.PyAudio()
# stream = p.open(format=pyaudio.paInt16,
#                 channels=CHANNELS,
#                 rate=RATE,
#                 input=True,
#                 input_device_index=12,  # your mic index
#                 frames_per_buffer=CHUNK)

# print("🎤 Listening...")

# last_spoken = ""
# last_time = 0

# def clean_text(text):
#     text = text.strip().lower()
#     text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
#     return text

# # Worker thread: runs Whisper on audio in queue
# def transcribe_worker():
#     global last_spoken, last_time
#     while True:
#         audio_np = audio_queue.get()
#         if audio_np is None:
#             break  # Stop signal
#         if np.abs(audio_np).mean() > THRESHOLD:
#             result = model.transcribe(
#                 audio_np,
#                 language='en',
#                 fp16=False,
#                 beam_size=5,
#                 best_of=5
#             )
#             text = clean_text(result["text"])
#             now = time.time()
#             if text and text != last_spoken and (now - last_time) > DEBOUNCE_INTERVAL:
#                 print("🗣️", result["text"].strip())
#                 last_spoken = text
#                 last_time = now
#         else:
#             print("...silent...")

# # Start transcription thread
# thread = threading.Thread(target=transcribe_worker, daemon=True)
# thread.start()

# try:
#     buffer = b''
#     while True:
#         chunk = stream.read(CHUNK, exception_on_overflow=False)
#         buffer += chunk
#         if len(buffer) >= RATE * 2 * BUFFER_DURATION:
#             # Convert to float32 PCM [-1, 1]
#             audio_np = np.frombuffer(buffer, np.int16).astype(np.float32) / 32768.0
#             buffer = b''
#             audio_queue.put(audio_np)
# except KeyboardInterrupt:
#     print("🛑 Stopping...")
#     audio_queue.put(None)  # signal thread to stop
#     thread.join()
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

# import asyncio
# import numpy as np
# import sounddevice as sd
# import time
# import re
# from whisper import load_model

# # Config
# RATE = 16000
# CHANNELS = 1
# THRESHOLD = 0.02
# BUFFER_DURATION = 2  # seconds
# DEBOUNCE_INTERVAL = 3  # seconds

# # Load Whisper model
# model = load_model("turbo", device="cuda")

# # Async queue for audio data
# audio_queue = asyncio.Queue()
# last_spoken = ""
# last_time = 0

# def clean_text(text):
#     text = text.strip().lower()
#     text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)
#     return text

# def audio_callback(indata, frames, time_info, status):
#     # Called in a background thread
#     if status:
#         print("Audio callback status:", status)
#     # Copy data to avoid pointer issues
#     audio_data = indata.copy()
#     # Send data to asyncio queue using main event loop
#     asyncio.run_coroutine_threadsafe(audio_queue.put(audio_data), loop)

# # Inside process_audio (replace the existing one)
# async def process_audio():
#     global last_spoken, last_time, speech_buffer

#     print("🎤 Listening...")

#     buffer_seconds = 3
#     sample_buffer = int(RATE * buffer_seconds)
#     audio_accum = np.zeros(sample_buffer, dtype=np.float32)
#     position = 0

#     while True:
#         chunk = await audio_queue.get()
#         audio_float = chunk.flatten().astype(np.float32) / 32768.0
#         chunk_len = len(audio_float)

#         if position + chunk_len <= sample_buffer:
#             audio_accum[position:position + chunk_len] = audio_float
#             position += chunk_len
#         else:
#             audio_accum = np.roll(audio_accum, -chunk_len)
#             audio_accum[-chunk_len:] = audio_float

#         # Process every second
#         if position >= RATE:  # 1 second
#             segment = audio_accum[-RATE*2:]  # last 2 seconds
#             result = model.transcribe(
#                 segment,
#                 language='en',
#                 fp16=False,
#                 beam_size=1,
#                 best_of=1
#             )
#             text = clean_text(result['text'])

#             if text and text != last_spoken:
#                 print("🗣️", result['text'].strip())
#                 last_spoken = text

# async def main():
#     print("🎤 Listening...")
#     sd.default.device = (12, None)  # (input_device_index, output_device_index)

#     with sd.InputStream(samplerate=RATE, channels=CHANNELS, callback=audio_callback, dtype='int16'):
#         await process_audio()

# # Get main event loop
# loop = asyncio.get_event_loop()

# try:
#     loop.run_until_complete(main())
# except KeyboardInterrupt:
#     print("🛑 Stopped")


## REPETITION

# import asyncio
# import numpy as np
# import sounddevice as sd
# import time
# import torch
# import re
# from whisper import load_model

# # Load VAD
# torch.set_num_threads(1)
# model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad')
# (get_speech_timestamps, _, _, _, _) = utils

# # Whisper config
# RATE = 16000
# CHANNELS = 1
# DEVICE_INDEX = 12
# CHUNK_DURATION = 0.5
# SILENCE_TIMEOUT = 1.5  # seconds of silence to consider speech ended

# # Load Whisper
# whisper_model = load_model("base.en", device="cuda")

# # State
# audio_queue = asyncio.Queue()
# transcribed_text = ""
# rolling_audio = []
# last_voice_time = 0
# in_speech = False

# def clean_text(text):
#     return re.sub(r'\s+', ' ', text.strip())

# def get_new_text(prev, current):
#     prev = prev.strip()
#     current = current.strip()

#     if not prev:
#         return current

#     max_overlap = 0
#     max_len = min(len(prev), len(current))

#     for i in range(max_len):
#         suffix = prev[i:]
#         if current.startswith(suffix):
#             max_overlap = len(suffix)
#             break

#     return current[max_overlap:].strip()

# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print("Status:", status)
#     asyncio.run_coroutine_threadsafe(audio_queue.put(indata.copy()), loop)

# async def process_audio():
#     global transcribed_text, rolling_audio, last_voice_time, in_speech
#     chunk_duration = 1.5  # seconds of audio per Whisper inference
#     buffer_size = int(RATE * chunk_duration)

#     print("Listening...")

#     while True:
#         chunk = await audio_queue.get()
#         audio_float = chunk.flatten().astype(np.float32) / 32768.0
#         audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

#         speech_timestamps = get_speech_timestamps(audio_tensor, model, sampling_rate=RATE)
#         now = time.time()

#         if speech_timestamps:
#             rolling_audio.append(audio_float)
#             last_voice_time = now
#             in_speech = True

#             full_audio = np.concatenate(rolling_audio)[-buffer_size:]
#             result = whisper_model.transcribe(
#                 full_audio,
#                 language='en',
#                 fp16=False,
#                 beam_size=5,
#                 best_of=5
#             )
#             new_text = clean_text(result["text"])
#             delta = get_new_text(transcribed_text, new_text)

#             if delta:
#                 transcribed_text += " " + delta
#                 print("\r" + transcribed_text.strip(), end="", flush=True)

#         else:
#             if in_speech and (now - last_voice_time) > SILENCE_TIMEOUT:
#                 print()  # newline to finalize current output
#                 transcribed_text = ""
#                 rolling_audio = []
#                 in_speech = False

# async def main():
#     sd.default.device = (DEVICE_INDEX, None)
#     with sd.InputStream(
#         samplerate=RATE,
#         channels=CHANNELS,
#         dtype='int16',
#         blocksize=int(RATE * CHUNK_DURATION),
#         callback=audio_callback
#     ):
#         await process_audio()

# loop = asyncio.get_event_loop()
# try:
#     loop.run_until_complete(main())
# except KeyboardInterrupt:
#     print("\n🛑 Stopped")

## WORKS

# import asyncio
# import numpy as np
# import sounddevice as sd
# import time
# import torch
# import re
# from whisper import load_model

# # Load Silero VAD from torch hub
# torch.set_num_threads(1)
# model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
# (get_speech_timestamps, _, _, _, _) = utils

# # Whisper config
# RATE = 16000
# CHANNELS = 1
# DEVICE_INDEX = 12
# CHUNK_DURATION = 0.5  # seconds

# # Whisper model
# whisper_model = load_model("turbo", device="cuda")

# # Global
# audio_queue = asyncio.Queue()
# speech_buffer = []
# last_text = ""
# last_time = 0
# transcribe_interval = 1.5  # seconds

# def clean_text(text):
#     text = text.strip().lower()
#     text = re.sub(r'\b(\w+)( \1\b)+', r'\1', text)  # remove repeated words
#     return text

# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print("Status:", status)
#     asyncio.run_coroutine_threadsafe(audio_queue.put(indata.copy()), loop)

# async def process_audio():
#     global speech_buffer, last_text, last_time

#     print("🎤 Listening...")
#     last_transcribed = time.time()

#     while True:
#         chunk = await audio_queue.get()
#         audio_float = chunk.flatten().astype(np.float32) / 32768.0
#         audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)

#         # Speech detection
#         speech = get_speech_timestamps(audio_tensor, model, sampling_rate=RATE)
#         if speech:
#             speech_buffer.append(audio_float)
#             now = time.time()

#             # Every transcribe_interval seconds, transcribe what's collected
#             if now - last_transcribed >= transcribe_interval:
#                 full_audio = np.concatenate(speech_buffer)
#                 result = whisper_model.transcribe(
#                     full_audio,
#                     language='en',
#                     fp16=False,
#                     beam_size=1,
#                     best_of=1
#                 )
#                 text = clean_text(result['text'])

#                 if text and text != last_text:
#                     print("🗣️", result['text'].strip())
#                     last_text = text

#                 last_transcribed = now
#         else:
#             if speech_buffer:
#                 # End of speech segment
#                 full_audio = np.concatenate(speech_buffer)
#                 result = whisper_model.transcribe(
#                     full_audio,
#                     language='en',
#                     fp16=False,
#                     beam_size=1,
#                     best_of=1
#                 )
#                 text = clean_text(result['text'])
#                 if text and text != last_text:
#                     print("🗣️", result['text'].strip())
#                     last_text = text

#                 speech_buffer = []

# async def main():
#     sd.default.device = (DEVICE_INDEX, None)
#     with sd.InputStream(
#         samplerate=RATE,
#         channels=CHANNELS,
#         dtype='int16',
#         blocksize=int(RATE * CHUNK_DURATION),
#         callback=audio_callback
#     ):
#         await process_audio()

# # Start
# loop = asyncio.get_event_loop()
# try:
#     loop.run_until_complete(main())
# except KeyboardInterrupt:
#     print("🛑 Stopped")

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
#     device_id = 12  # your specific input device ID

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
