
# from ollama import chat, ChatResponse
# from RTTTS import stream_speech, say_chunk, finish_speech
# from RTSTT import start_transcription
# import time
# from difflib import SequenceMatcher

# # Track what the AI just said
# recent_ai_responses = []
# TTS_COOLDOWN_TIME = 5  # seconds after TTS finishes
# last_tts_finish_time = 0

# def similarity(a, b):
#     """Calculate similarity between two strings (0-1, where 1 is identical)"""
#     return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()

# def is_likely_feedback(transcript):
#     """Check if transcript is likely feedback from our own TTS"""
#     global last_tts_finish_time
    
#     # If we just finished speaking, ignore input for a few seconds
#     if time.time() - last_tts_finish_time < TTS_COOLDOWN_TIME:
#         print(f"[DEBUG] Ignoring input during cooldown: '{transcript}'")
#         return True
    
#     # Check if transcript is similar to recent AI responses
#     for ai_response in recent_ai_responses:
#         if similarity(transcript, ai_response) > 0.7:  # 70% similarity threshold
#             print(f"[DEBUG] Detected feedback - ignoring: '{transcript}'")
#             return True
    
#     return False

# def send_to_ollama(transcript: str) -> None:
#     global recent_ai_responses, last_tts_finish_time
    
#     if not transcript.strip():
#         return
    
#     # Check if this is likely feedback from our own TTS
#     if is_likely_feedback(transcript):
#         return
    
#     print("\n-- Silence detected, sending transcript to Ollama --\n")
    
#     stream = chat(
#         model="Dolphin-MIO",
#         messages=[{'role': 'admin', 'content': transcript.strip()}],
#         stream=True,
#     )
    
#     print("Ollama response: ", end='', flush=True)
    
#     stream_speech()
    
#     buffer = ""
#     full_response = ""  # Track the complete AI response
    
#     for chunk in stream:
#         content = chunk['message']['content']
#         print(content, end='', flush=True)
#         buffer += content
#         full_response += content  # Build complete response
        
#         # Speak only when we detect end of sentence or buffer is long
#         if any(buffer.endswith(p) for p in [".", "!", "?", "\n"]) or len(buffer) > 100:
#             say_chunk(buffer.strip())
#             buffer = ""
    
#     # Say any remaining buffered content
#     if buffer.strip():
#         print(f"\n[DEBUG] Flushing remaining to TTS: {buffer.strip()}\n")
#         say_chunk(buffer.strip())
#         full_response += buffer
    
#     print("\n-- End of Ollama response --")
    
#     # Wait for TTS to fully finish
#     finish_speech()
    
#     # Record the AI's response and when TTS finished
#     recent_ai_responses.append(full_response.strip())
#     last_tts_finish_time = time.time()
    
#     # Keep only the last few responses to avoid memory buildup
#     if len(recent_ai_responses) > 3:
#         recent_ai_responses.pop(0)
    
#     print(f"[DEBUG] TTS finished, recorded response for feedback detection")

# if __name__ == "__main__":
#     start_transcription(send_to_ollama)

# ollama_te.py (relevant "ALSO WORKING" snippet)


##WORKING
# from ollama import chat # ChatResponse is not used in this version of send_to_ollama
# from RTTTS import stream_speech, say_chunk, finish_speech # tts_done is used by RTSTT internally
# from RTSTT import start_transcription
# import time
# from difflib import SequenceMatcher
#
#
# recent_ai_responses = []
# TTS_COOLDOWN_TIME = 2  
# last_tts_finish_time = 0
#
# def similarity(a, b):
#     return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()
#
# def is_likely_feedback(transcript):
#     global last_tts_finish_time
#
#     if time.time() - last_tts_finish_time < TTS_COOLDOWN_TIME:
#         print(f"[OLLAMA_TE DEBUG] Ignoring input during TTS cooldown: '{transcript}'")
#         return True
#
#     for ai_response in recent_ai_responses:
#         if similarity(transcript, ai_response) > 0.5:
#             print(f"[OLLAMA_TE DEBUG] Detected feedback (similarity) - ignoring: '{transcript}'")
#             return True
#     return False
#
# def send_to_ollama(transcript: str) -> None:
#     global recent_ai_responses, last_tts_finish_time
#
#     if not transcript.strip():
#         return
#
#     if is_likely_feedback(transcript):
#         return
#
#     print(f"\n[USER TRANSCRIPT] {transcript}") 
#     print("-- Sending transcript to Ollama --")
#
#
#     messages = [{'role': 'user', 'content': transcript.strip()}]
#
#     try:
#         stream = chat(
#             model="glados", 
#             messages=messages,
#             stream=True,
#         )
#     except Exception as e:
#         print(f"[OLLAMA ERROR] Failed to get response from Ollama: {e}")
#         return 
#
#     print("[OLLAMA RESPONSE]: ", end='', flush=True)
#
#     stream_speech() 
#
#     buffer = ""
#     full_response = ""
#
#     for chunk in stream:
#         content = chunk['message']['content']
#         print(content, end='', flush=True)
#         buffer += content
#         full_response += content
#
#         if any(buffer.endswith(p) for p in [".", "!", "?", "\n"]) or len(buffer) > 100:
#             say_chunk(buffer.strip()) 
#             buffer = ""
#
#     if buffer.strip():
#
#         say_chunk(buffer.strip())
#
#
#     print("\n-- End of Ollama response --")
#
#     finish_speech() 
#
#     recent_ai_responses.append(full_response.strip())
#     last_tts_finish_time = time.time() 
#
#     if len(recent_ai_responses) > 3: 
#         recent_ai_responses.pop(0)
#
#
#
# if __name__ == "__main__":
#     print("🎙️ Speak to interact (Ctrl+C to stop)...")
#     try:
#         start_transcription(send_to_ollama)
#     except KeyboardInterrupt:
#         print("\n🛑 Exiting application.")
#     except Exception as e:
#         print(f"\n💥 An unexpected error occurred: {e}")
#
#
#
from ollama import chat
from RTTTS import stream_speech, say_chunk, finish_speech
from RTSTT import start_transcription, mic_paused_event
import time
import threading

def send_to_ollama(transcript: str) -> None:
    if not transcript.strip():
        return
    
    print(f"\n[USER TRANSCRIPT] {transcript}") 
    print("-- Sending transcript to Ollama --")
    
    # Pause microphone before AI speaks
    mic_paused_event.set()
    print("[DEBUG] Microphone paused for TTS")
    
    # Clear any remaining audio from the queue to prevent old audio from being processed
    import RTSTT
    while not RTSTT.q.empty():
        try:
            RTSTT.q.get_nowait()
        except:
            break

    messages = [{'role': 'user', 'content': transcript.strip()}]
    
    try:
        stream = chat(
            model="glados", 
            messages=messages,
            stream=True,
        )
    except Exception as e:
        print(f"[OLLAMA ERROR] Failed to get response from Ollama: {e}")
        # Ensure microphone is unpaused even on error
        mic_paused_event.clear()
        print("[DEBUG] Microphone unpaused after error")
        return 

    print("[OLLAMA RESPONSE]: ", end='', flush=True)
    
    stream_speech() 
    
    buffer = ""
    
    for chunk in stream:
        content = chunk['message']['content']
        print(content, end='', flush=True)
        buffer += content
        
        if any(buffer.endswith(p) for p in [".", "!", "?", "\n"]) or len(buffer) > 100:
            say_chunk(buffer.strip()) 
            buffer = ""
    
    if buffer.strip():
        say_chunk(buffer.strip())
  
    print("\n-- End of Ollama response --")
    
    # Wait for TTS to finish completely
    finish_speech()
    
    # Add a small delay to ensure TTS audio has finished playing
    time.sleep(0.5)
    
    # Clear any audio that might have been captured during TTS
    while not RTSTT.q.empty():
        try:
            RTSTT.q.get_nowait()
        except:
            break
    
    # Unpause microphone after TTS finishes
    mic_paused_event.clear()
    print("[DEBUG] Microphone unpaused after TTS finished")

if __name__ == "__main__":
    print("🎙️ Speak to interact (Ctrl+C to stop)...")
    try:
        # Ensure microphone starts unpaused
        mic_paused_event.clear()
        print(f"[INITIAL DEBUG] Microphone starts unpaused: {not mic_paused_event.is_set()}")

        start_transcription(send_to_ollama)

    except KeyboardInterrupt:
        print("\n🛑 Exiting application.")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
