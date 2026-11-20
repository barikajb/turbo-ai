
import pyttsx3

engine = pyttsx3.init("sapi5")

voices = engine.getProperty("voices")

print("\nAvailable voices on your computer:\n")

for index, voice in enumerate(voices):
    print(f"Voice {index}: {voice.name}")
    print(f"ID: {voice.id}")
    print("-" * 40)

engine.stop()
