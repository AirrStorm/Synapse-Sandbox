from ollama import chat
from RTTTS import stream_speech, say_chunk, finish_speech
from RTSTT import mic_paused_event
import time

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

