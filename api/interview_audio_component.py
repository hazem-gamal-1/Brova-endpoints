import os
from io import BytesIO
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()
voices = {"male": "JBFqnCBsd6RMkjVDRZzb", "female": "FGY2WhTYpPnrIDTdsKH5"}


class InterviewAudioComponent:
    def __init__(self, gender):
        self.elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        self.voice_id = voices[gender]
        self.model_id = "eleven_multilingual_v2"

    def convert_text_to_speech(self, text: str) -> bytes:
        try:
            audio_stream = self.elevenlabs.text_to_speech.stream(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )

            # Collect all streamed audio bytes
            audio_bytes = b"".join(
                chunk for chunk in audio_stream if isinstance(chunk, bytes)
            )
            return audio_bytes
        except Exception as e:
            raise Exception(f"Text to speech conversion failed: {str(e)}")

    def stream_text_to_speech(self, text: str):
        """Stream audio chunks as they're generated"""
        try:
            audio_stream = self.elevenlabs.text_to_speech.stream(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )

            # Yield chunks as they arrive
            for chunk in audio_stream:
                if isinstance(chunk, bytes):
                    yield chunk
        except Exception as e:
            raise Exception(f"Text to speech streaming failed: {str(e)}")
