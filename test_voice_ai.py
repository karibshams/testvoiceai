import os
from io import BytesIO
import pyaudio
import wave
import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
MALE_VOICE_ID = os.getenv("MALE_VOICE_ID")
FEMALE_VOICE_ID = os.getenv("FEMALE_VOICE_ID")

# Supported languages
LANGUAGES = ["english", "hindi", "urdu", "portuguese"]

client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------
# RECORD AUDIO FUNCTION
# -----------------------
def record_audio():
    """Record audio from microphone until Ctrl+C is pressed"""
    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=16000,
        input=True,
        frames_per_buffer=1024
    )
    
    print("🎤 Recording... Press Ctrl+C to stop")
    frames = []
    try:
        while True:
            frames.append(stream.read(1024))
    except KeyboardInterrupt:
        print("\n✓ Recording stopped")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    # Convert to WAV format
    audio_data = BytesIO()
    with wave.open(audio_data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(16000)
        wav_file.writeframes(b"".join(frames))
    
    audio_data.seek(0)
    return audio_data

# -----------------------
# ELEVENLABS STT FUNCTION
# -----------------------
def speech_to_text(audio_data):
    """Convert speech to text using ElevenLabs STT API"""
    audio_data.seek(0)
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": ("audio.wav", audio_data, "audio/wav")}
    data = {"model_id": "scribe_v1"}
    
    print("📝 Transcribing audio...")
    response = requests.post(url, headers=headers, files=files, data=data)
    
    if response.status_code != 200:
        print(f"❌ ElevenLabs STT Error: {response.status_code}")
        print(f"Response: {response.text}")
        return ""
    
    result = response.json()
    return result.get("text", "")

# -----------------------
# OPENAI RESPONSE
# -----------------------
def generate_response(prompt, language):
    """Generate AI response in the specified language"""
    lang_map = {
        "english": "English",
        "hindi": "Hindi",
        "urdu": "Urdu",
        "portuguese": "Portuguese"
    }
    
    target_lang = lang_map.get(language, "English")
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": f"You are a helpful assistant. Always respond in {target_lang}."
        }, {
            "role": "user",
            "content": prompt
        }],
        temperature=0.7,
    )
    
    return response.choices[0].message.content

# -----------------------
# ELEVENLABS TTS FUNCTION
# -----------------------
def text_to_speech(text, gender):
    """Convert text to speech using ElevenLabs TTS API"""
    # Select voice based on gender
    voice_id = MALE_VOICE_ID if gender == "male" else FEMALE_VOICE_ID
    
    if not voice_id:
        print(f"❌ Voice ID not found for {gender}")
        return
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.75,
            "style": 0.0,
            "use_speaker_boost": True
        }
    }
    
    print("🔊 Generating speech...")
    response = requests.post(url, headers=headers, json=payload, stream=True)
    
    if response.status_code != 200:
        print(f"❌ ElevenLabs TTS Error: {response.status_code}")
        print(f"Response: {response.text}")
        return
    
    # Save audio to a file for debugging
    with open("output_audio.wav", "wb") as audio_file:
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                audio_file.write(chunk)
    
    # Play audio
    p = pyaudio.PyAudio()
    try:
        stream = p.open(
            format=pyaudio.paInt16,  # Ensure format matches ElevenLabs output
            channels=1,             # Mono audio
            rate=22050,             # Match ElevenLabs sample rate
            output=True
        )
        
        print("🎵 Playing audio...")
        for chunk in response.iter_content(chunk_size=4096):
            if chunk:
                stream.write(chunk)
    except Exception as e:
        print(f"❌ Audio playback error: {str(e)}")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# -----------------------
# MAIN FUNCTION
# -----------------------
def main():
    print("=" * 50)
    print("🌍 Multi-Language Voice AI Assistant")
    print("=" * 50)
    
    # Check API keys
    if not OPENAI_API_KEY or not ELEVENLABS_API_KEY:
        print("❌ Error: API keys not found in .env file")
        return
    
    if not MALE_VOICE_ID or not FEMALE_VOICE_ID:
        print("❌ Error: Voice IDs not found in .env file")
        return
    
    # Language selection
    print(f"\nAvailable languages: {', '.join(LANGUAGES)}")
    language = input("Select language: ").lower().strip()
    
    if language not in LANGUAGES:
        print(f"❌ Invalid language. Choose from: {', '.join(LANGUAGES)}")
        return
    
    # Gender selection
    gender = input("Select voice gender (male/female): ").lower().strip()
    
    if gender not in ["male", "female"]:
        print("❌ Invalid gender. Choose 'male' or 'female'")
        return
    
    # Input method selection
    print("\n" + "=" * 50)
    print("Choose input method:")
    print("1. Record audio (STT)")
    print("2. Type text")
    print("=" * 50)
    choice = input("Enter choice (1/2): ").strip()
    
    user_input = ""
    
    if choice == "1":
        # Record and transcribe
        audio = record_audio()
        user_input = speech_to_text(audio)
        
        if not user_input:
            print("❌ Failed to transcribe audio")
            return
        
        print(f"\n📖 Transcribed text: {user_input}")
    
    elif choice == "2":
        # Text input
        user_input = input("\nEnter your text: ").strip()
        
        if not user_input:
            print("❌ No input provided")
            return
    
    else:
        print("❌ Invalid choice")
        return
    
    # Generate AI response
    print("\n🤖 Generating AI response...")
    ai_response = generate_response(user_input, language)
    
    print("\n" + "=" * 50)
    print(f"AI Response ({language.title()}):")
    print("=" * 50)
    print(ai_response)
    print("=" * 50)
    
    # Convert to speech
    text_to_speech(ai_response, gender)
    
    print("\n✅ Done!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")