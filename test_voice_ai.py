import os
import sys
from io import BytesIO
import pyaudio
import wave
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

VOICES = {
    "male_en": "cgSgspJ2msm3yXmAZjWA",
    "female_en": "21m00Tcm4TlvDq8ikWAM",
    "male_hi": "nPczCjzI2devNBz1zQrb",
    "female_hi": "EXAVITQu4vr4xnSDxMaL",
    "male_pt": "onwK4e9ZhZ08zL12Z3AQ",
    "female_pt": "9BWtsMINqrJLrRacOk9Q",
}

LANGUAGES = ["english", "hindi", "portugese"]

client = OpenAI(api_key=OPENAI_API_KEY)


def record_audio(duration=5):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)
    
    print("🎤 Recording... Press Ctrl+C to stop")
    frames = []
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        print("✓ Recording stopped")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_data = BytesIO()
    with wave.open(audio_data, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wav_file.setframerate(16000)
        wav_file.writeframes(b"".join(frames))
    
    audio_data.seek(0)
    return audio_data


def speech_to_text(audio_data):
    url = "https://api.elevenlabs.io/v1/scribe"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    files = {"file": audio_data}
    
    response = requests.post(url, headers=headers, files=files)
    response.raise_for_status()
    return response.json().get("transcript", "")


def generate_response(prompt, language):
    lang_names = {"english": "English", "hindi": "Hindi", "portugese": "Portuguese"}
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": f"Respond in {lang_names[language]}. {prompt}"}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def text_to_speech(text, language, gender):
    voice_key = f"{gender}_{language[0:2].lower()}"
    voice_id = VOICES.get(voice_key)
    
    if not voice_id:
        print(f"❌ Voice not available: {voice_key}")
        return
    
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    
    response = requests.post(url, headers=headers, json=payload, stream=True)
    response.raise_for_status()
    
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
    
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            stream.write(chunk)
    
    stream.stop_stream()
    stream.close()
    p.terminate()


def main():
    print("🌍 Multi-Language Voice AI\n")
    
    language = input(f"Select language {LANGUAGES}: ").lower()
    if language not in LANGUAGES:
        print("❌ Invalid language")
        return
    
    gender = input("Select voice (male/female): ").lower()
    if gender not in ["male", "female"]:
        print("❌ Invalid gender")
        return
    
    print("\n1. Record & Transcribe")
    print("2. Text Input")
    choice = input("Choose (1/2): ")
    
    if choice == "1":
        audio = record_audio()
        user_input = speech_to_text(audio)
        print(f"📖 Transcribed: {user_input}")
    else:
        user_input = input("Enter text: ")
    
    print("🤖 Generating response...")
    ai_response = generate_response(user_input, language)
    print(f"Response: {ai_response}\n")
    
    print(f"🔊 Speaking in {gender} {language}...")
    text_to_speech(ai_response, language, gender)
    print("✓ Done")


if __name__ == "__main__":
    main()