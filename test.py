#
# import sounddevice as sd
#
# devices = sd.query_devices()
# for i, device in enumerate(devices):
#     if device['max_input_channels'] > 0:
#         print(f"{i}: {device['name']} ({device['hostapi']}) - {device['max_input_channels']} in")


import sounddevice as sd

device_id = 12
test_rates = [8000, 11025, 16000, 22050, 32000, 44100, 48000]

print(f"Testing sample rates for device {device_id}...\n")
for rate in test_rates:
    try:
        sd.check_input_settings(device=device_id, samplerate=rate)
        print(f"✔️  {rate} Hz is supported")
    except Exception as e:
        print(f"❌  {rate} Hz not supported: {e}")
