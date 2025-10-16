import os
import requests
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"  # Rachel (changeable)

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_response(prompt: str) -> str:
    """Get text response from OpenAI."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    return response.choices[0].message.content


def text_to_speech(text: str, output_file: str = "output.mp3") -> str:
    """Convert text to speech using ElevenLabs TTS."""
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    payload = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }

    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()

    with open(output_file, "wb") as f:
        f.write(response.content)

    return output_file


def speech_to_text(audio_file: str) -> str:
    """Transcribe audio using ElevenLabs Scribe (STT)."""
    url = "https://api.elevenlabs.io/v1/scribe"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}

    with open(audio_file, "rb") as f:
        files = {"file": f}
        response = requests.post(url, headers=headers, files=files)

    response.raise_for_status()
    return response.json().get("transcript", "")


def test_voice_flow():
    """Test the complete voice AI flow."""
    print("🎤 Voice AI Testing Suite\n")

    # Step 1: Text → OpenAI Response
    user_input = "Tell me a fun fact about space in one sentence."
    print(f"📝 User Input: {user_input}")

    ai_response = generate_response(user_input)
    print(f"🤖 OpenAI Response: {ai_response}\n")

    # Step 2: Response → TTS
    print("🔊 Converting to speech...")
    audio_file = text_to_speech(ai_response, "response.mp3")
    print(f"✅ Audio saved: {audio_file}\n")

    # Step 3 (Optional): STT Round-trip
    print("🔙 Testing STT (Scribe)...")
    transcribed_text = speech_to_text(audio_file)
    print(f"📖 Transcribed: {transcribed_text}\n")

    # Step 4 (Optional): Feed transcribed text back to OpenAI
    if transcribed_text:
        followup = generate_response(
            f"The user said: '{transcribed_text}'. Confirm you understood them."
        )
        print(f"🤖 Confirmation: {followup}\n")

    print("✨ Voice flow test complete!")


if __name__ == "__main__":
    test_voice_flow()