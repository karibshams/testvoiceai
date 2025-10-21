from elevenlabs import ElevenLabs
from elevenlabs.audio import play

client = ElevenLabs(
    api_key="YOUR_API_KEY"
)

audio = client.text_to_speech.convert(
    text="The first move is what sets everything in motion.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)