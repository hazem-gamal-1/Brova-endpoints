import os
from io import BytesIO
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

load_dotenv()
voices = {"male": "JBFqnCBsd6RMkjVDRZzb", "female": "FGY2WhTYpPnrIDTdsKH5"}


class InterviewAudioComponent:
    def __init__(self, gender):
        self.elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))
        self.voice_id = voices[gender]
        self.model_id = "eleven_multilingual_v2"

    def convert_speech_to_text(self, audio_file_path: str) -> str:
        try:
            with open(audio_file_path, "rb") as f:
                audio_data = BytesIO(f.read())

                transcription = self.elevenlabs.speech_to_text.convert(
                    file=audio_data,
                    model_id="scribe_v1",
                    tag_audio_events=True,
                    language_code=None,
                    diarize=True,
                )

                return transcription.text
        except Exception as e:
            raise Exception(f"Speech to text conversion failed: {str(e)}")

    def convert_text_to_speech(self, text: str) -> bytes:
        try:
            audio = self.elevenlabs.text_to_speech.convert(
                text=text,
                voice_id=self.voice_id,
                model_id=self.model_id,
                output_format="mp3_44100_128",
            )

            # Convert generator to bytes
            audio_bytes = b"".join(audio)
            return audio_bytes
        except Exception as e:
            raise Exception(f"Text to speech conversion failed: {str(e)}")
