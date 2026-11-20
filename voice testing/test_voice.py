
import pyttsx3

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 170)
engine.setProperty("volume", 1.0)

print("Testing TURBO voice...")
engine.say("Hello! This is Turbo. Can you hear me?")
engine.runAndWait()

print("Voice test finished.")
