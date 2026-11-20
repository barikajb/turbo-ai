
import json
import pyaudio
from vosk import Model, KaldiRecognizer

# Load the downloaded offline speech model
MODEL_PATH = "vosk-model-small-en-us-0.15"
model = Model(MODEL_PATH)

# Configure the microphone
RATE = 16000
CHUNK = 4000

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

recognizer = KaldiRecognizer(model, RATE)

print("TURBO offline voice test is ready.")
print("Speak a sentence. Press Ctrl+C to stop.")

try:
    while True:
        data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            text = result.get("text", "")

            if text:
                print("You said:", text)

except KeyboardInterrupt:
    print("\nOffline voice test stopped.")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()