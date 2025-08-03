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

def wait_for_wake_word(model_path, mic_index, access_key):
    """
    Blocks execution until wake word is detected.
    """
    porcupine = pvporcupine.create(
        access_key=access_key,
        keyword_paths=[model_path]
    )

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        input_device_index=mic_index,
        frames_per_buffer=porcupine.frame_length
    )

    print("🎤 Listening for 'Hey MIO'...")

    try:
        while True:
            pcm = stream.read(porcupine.frame_length, exception_on_overflow=False)
            pcm_unpacked = struct.unpack_from("h" * porcupine.frame_length, pcm)

            if porcupine.process(pcm_unpacked) >= 0:
                print("✅ Wake word detected: Hey MIO!")
                return True

    except KeyboardInterrupt:
        print("🛑 Stopped")
        return False

    finally:
        stream.close()
        pa.terminate()
        porcupine.delete()

