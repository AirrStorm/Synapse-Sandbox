# import pvporcupine
# import pyaudio
# import struct
#
# # Path to your downloaded "Hey Jarvis" .ppn file
# MODEL_PATH = "voices/Hey-MIO.ppn"
#
# porcupine = pvporcupine.create(access_key='KcZFoUcTWB0a8mZaciQRWxwE3/LQM1hQAdc5XWLnl/NIDLtug1LbwA==', keyword_paths=[MODEL_PATH])
#
# pa = pyaudio.PyAudio()
# stream = pa.open(
#     rate=porcupine.sample_rate,
#     channels=1,
#     format=pyaudio.paInt16,
#     input=True,
#     input_device_index= 12,
#     frames_per_buffer=porcupine.frame_length
# )
#
# print("🎤 Listening for 'Hey MIO'...")
#
# try:
#     while True:
#         pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
#         pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)
#
#         result = porcupine.process(pcm_unpacked)
#         if result >= 0:
#             print("✅ Wake word detected: Hey MIO!")
#
# except KeyboardInterrupt:
#     print("🛑 Stopped")
#
# finally:
#     stream.close()
#     pa.terminate()
#     porcupine.delete()


import pvporcupine
import pyaudio
import struct
import os
import subprocess

# Path to your downloaded "Hey MIO" .ppn file
MODEL_PATH = "voices/Hey-MIO.ppn"

porcupine = pvporcupine.create(
    access_key='KcZFoUcTWB0a8mZaciQRWxwE3/LQM1hQAdc5XWLnl/NIDLtug1LbwA==',
    keyword_paths=[MODEL_PATH]
)

pa = pyaudio.PyAudio()
stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    input_device_index=12,  # your mic index
    frames_per_buffer=porcupine.frame_length
)

print("🎤 Listening for 'Hey MIO'...")

try:
    while True:
        pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
        pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

        result = porcupine.process(pcm_unpacked)
        if result >= 0:
            print("✅ Wake word detected: Hey MIO!")

            # Run your main.py
            subprocess.run(["python", "main.py"])  # or os.system("python main.py")

            # Exit after running main.py
            break

except KeyboardInterrupt:
    print("🛑 Stopped")

finally:
    stream.close()
    pa.terminate()
    porcupine.delete()
