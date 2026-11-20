
import pyaudio
import numpy as np
from openwakeword.model import Model

# Load the pretrained wake-word model
model = Model(
    wakeword_models=["hey_jarvis"],
    inference_framework="onnx"
)

# Audio settings
RATE = 16000
CHUNK = 1280

audio = pyaudio.PyAudio()

stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    frames_per_buffer=CHUNK
)

print("TURBO wake-word test is running.")
print("Say: Hey Jarvis")
print("Press Ctrl+C to stop.")

try:
    while True:
        audio_data = stream.read(
            CHUNK,
            exception_on_overflow=False
        )

        frame = np.frombuffer(
            audio_data,
            dtype=np.int16
        )

        predictions = model.predict(frame)

        for wake_word, score in predictions.items():
            if float(score) > 0.5:
                print(
                    f"Wake word detected: {wake_word} "
                    f"(score: {float(score):.2f})"
                )

except KeyboardInterrupt:
    print("\nTest stopped.")

finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
