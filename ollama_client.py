#!/usr/bin/env python3
"""
Ollama Client for AI Chat Integration
Handles communication between speech input and AI model
"""

import time
from typing import Optional
import threading
from ollama import chat
from RTTTS import stream_speech, say_chunk, finish_speech, cleanup_tts
from RTSTT import pause_microphone, resume_microphone, clear_audio_buffer

# Conversation context (optional - for multi-turn conversations)
conversation_history = []
MAX_HISTORY_LENGTH = 10  # Keep last 10 exchanges

# Processing state
processing_lock = threading.Lock()
is_processing = False

def test_ollama_connection() -> bool:
    """Test if Ollama is running and the model is available"""
    try:
        print("[OLLAMA] Testing connection...")
        response = chat(
            model="mistral",
            messages=[{'role': 'user', 'content': 'Hi'}],
            stream=False,
            options={'num_predict': 10}  # Short response for testing
        )
        print("[OLLAMA] ✅ Connection successful")
        return True
    except Exception as e:
        print(f"[OLLAMA ERROR] Connection failed: {e}")
        print("Make sure Ollama is running and 'mistral' model is installed:")
        print("  ollama pull mistral")
        return False

def add_to_history(role: str, content: str) -> None:
    """Add message to conversation history with length management"""
    global conversation_history
    
    if not content.strip():
        return
    
    conversation_history.append({'role': role, 'content': content.strip()})
    
    # Keep history manageable
    if len(conversation_history) > MAX_HISTORY_LENGTH * 2:  # *2 because user+assistant pairs
        conversation_history = conversation_history[-MAX_HISTORY_LENGTH * 2:]

def get_conversation_context() -> list:
    """Get conversation history for context-aware responses"""
    return conversation_history.copy()

def handle_special_commands(transcript: str) -> bool:
    """
    Handle special voice commands
    Returns True if command was handled, False otherwise
    """
    transcript_lower = transcript.lower().strip()
    
    if transcript_lower in ["clear history", "clear conversation", "start over", "new conversation"]:
        clear_conversation_history()
        say_chunk("Conversation history cleared. Starting fresh!")
        return True
    
    elif transcript_lower in ["show history", "conversation summary", "what did we talk about"]:
        summary = get_conversation_summary()
        print(f"[HISTORY]\n{summary}")
        say_chunk("Here's a summary of our conversation so far.")
        return True
    
    elif transcript_lower in ["stop talking", "be quiet", "shut up"]:
        # This is a bit tricky since we're already in the ollama processing
        # We can at least not send a response
        say_chunk("Okay, I'll be quiet.")
        return True
    
    return False

def process_with_ai(transcript: str) -> str:
    """
    Process transcript with AI and return response text
    Separated from TTS for better error handling
    """
    try:
        # Prepare messages (context-aware conversation)
        messages = get_conversation_context()
        messages.append({'role': 'user', 'content': transcript})
        
        ai_response = ""
        
        # Get streaming response from Ollama
        stream = chat(
            model="mistral", 
            messages=messages,
            stream=True,
            options={
                'temperature': 0.7,  # Adjust creativity
                'top_p': 0.9,       # Nucleus sampling
                'num_predict': 512,  # Max response length
                'stop': ['<|endoftext|>', '</s>']  # Stop tokens
            }
        )
        
        # Collect response
        for chunk in stream:
            try:
                content = chunk['message']['content']
                ai_response += content
            except KeyError:
                print(f"[DEBUG] Unexpected chunk format: {chunk}")
                continue
        
        return ai_response.strip()
        
    except Exception as e:
        print(f"[OLLAMA ERROR] {e}")
        error_message = "I'm sorry, I'm having trouble connecting to the AI service right now."
        return error_message

def send_to_ollama(transcript: str) -> None:
    """
    Send transcript to Ollama and handle TTS response
    
    Args:
        transcript: The transcribed speech text
    """
    global is_processing
    
    # Prevent concurrent processing
    with processing_lock:
        if is_processing:
            print("[OLLAMA] Already processing, skipping...")
            return
        is_processing = True
    
    try:
        if not transcript or not transcript.strip():
            print("[DEBUG] Empty transcript, skipping...")
            return
        
        transcript = transcript.strip()
        print(f"\n[USER] {transcript}")
        
        # Check for special commands first
        if handle_special_commands(transcript):
            return
        
        print("🤖 Processing with AI...")
        
        # Pause microphone and clear any buffered audio
        pause_microphone()
        clear_audio_buffer()
        
        # Add user message to history
        add_to_history('user', transcript)
        
        # Get AI response
        ai_response = process_with_ai(transcript)
        
        if not ai_response:
            print("[OLLAMA] No response generated")
            return
        
        print(f"[AI] {ai_response}")
        
        # Handle TTS response with streaming
        try:
            # Start TTS streaming
            stream_speech()
            
            # Split response into chunks for better streaming
            sentences = split_into_sentences(ai_response)
            
            for sentence in sentences:
                if sentence.strip():
                    say_chunk(sentence.strip())
            
            # Add AI response to history
            add_to_history('assistant', ai_response)
            
        except Exception as tts_error:
            print(f"[TTS ERROR] {tts_error}")
            # Still add to history even if TTS fails
            add_to_history('assistant', ai_response)
        
        finally:
            # Always ensure TTS cleanup
            try:
                finish_speech()
            except:
                pass
        
    except Exception as e:
        print(f"[OLLAMA ERROR] Unexpected error: {e}")
        
        # Provide fallback response
        try:
            error_message = "I encountered an error while processing your request."
            say_chunk(error_message)
        except:
            pass
        
    finally:
        # Always ensure cleanup happens
        try:
            # Small delay to ensure audio playback is complete
            time.sleep(0.3)
            
            # Clear any audio captured during AI response
            clear_audio_buffer()
            
            # Resume microphone
            resume_microphone()
            print("🎤 Ready for next input...\n")
            
        except Exception as cleanup_error:
            print(f"[CLEANUP ERROR] {cleanup_error}")
            # Force resume microphone even if other cleanup fails
            try:
                resume_microphone()
            except:
                pass
        
        finally:
            # Reset processing flag
            with processing_lock:
                is_processing = False

def split_into_sentences(text: str) -> list:
    """
    Split text into sentences for better TTS streaming
    """
    if not text:
        return []
    
    # Simple sentence splitting - can be improved
    import re
    
    # Split on sentence endings, but keep the punctuation
    sentences = re.split(r'([.!?]+)', text)
    
    # Recombine sentences with their punctuation
    result = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence = sentences[i] + sentences[i + 1]
            if sentence.strip():
                result.append(sentence.strip())
        else:
            if sentences[i].strip():
                result.append(sentences[i].strip())
    
    # If no sentences were found, return the original text
    if not result and text.strip():
        result = [text.strip()]
    
    return result

def clear_conversation_history() -> None:
    """Clear the conversation history"""
    global conversation_history
    conversation_history = []
    print("[DEBUG] Conversation history cleared")

def get_conversation_summary() -> str:
    """Get a summary of the current conversation"""
    if not conversation_history:
        return "No conversation history available."
    
    summary = f"Conversation has {len(conversation_history)} messages:\n"
    for i, msg in enumerate(conversation_history[-6:], 1):  # Show last 6 messages
        role_icon = "👤" if msg['role'] == 'user' else "🤖"
        content_preview = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
        summary += f"{i}. {role_icon} {content_preview}\n"
    
    return summary

def shutdown_ollama_client():
    """Cleanup function for graceful shutdown"""
    global is_processing
    
    print("[OLLAMA] Shutting down...")
    
    # Wait for any ongoing processing to complete
    timeout = 5.0
    start_time = time.time()
    
    while is_processing and (time.time() - start_time) < timeout:
        print("[OLLAMA] Waiting for processing to complete...")
        time.sleep(0.1)
    
    if is_processing:
        print("[OLLAMA] Warning: Forced shutdown while processing")
    
    # Cleanup TTS
    try:
        cleanup_tts()
    except Exception as e:
        print(f"[OLLAMA] TTS cleanup error: {e}")
    
    print("[OLLAMA] Shutdown complete")

# Enhanced send_to_ollama that handles special commands (alternative interface)
def send_to_ollama_enhanced(transcript: str) -> None:
    """Enhanced version that handles special commands - alias for send_to_ollama"""
    send_to_ollama(transcript)

# Test function
def test_ai_pipeline():
    """Test the complete AI pipeline"""
    print("[TEST] Testing AI pipeline...")
    
    if not test_ollama_connection():
        return False
    
    # Test with a simple query
    test_transcript = "Hello, can you hear me?"
    print(f"[TEST] Testing with: '{test_transcript}'")
    
    try:
        send_to_ollama(test_transcript)
        print("[TEST] ✅ AI pipeline test completed")
        return True
    except Exception as e:
        print(f"[TEST ERROR] {e}")
        return False

if __name__ == "__main__":
    # Test the ollama client
    test_ai_pipeline()