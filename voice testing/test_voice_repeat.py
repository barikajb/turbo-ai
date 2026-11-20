
import subprocess
import sys

def speak(message):
    script = """
import pyttsx3
import sys

engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 175)
engine.setProperty("volume", 1.0)
engine.say(sys.argv[1])
engine.runAndWait()
engine.stop()
"""

    result = subprocess.run(
        [sys.executable, "-c", script, message],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("Voice error:", result.stderr)


print("TURBO voice test starting...")

speak("TURBO is online.")
speak("This is the second voice test.")

print("TURBO voice test finished.")
