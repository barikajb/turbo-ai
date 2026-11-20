
import os
import subprocess
import webbrowser
from urllib.parse import quote_plus


# ============================================================
# TURBO COMPUTER CONTROL
# Windows application, website, and folder commands
# ============================================================

# 1. Websites TURBO is allowed to open directly
WEBSITES = {
    "youtube": "https://www.youtube.com",
    "google": "https://www.google.com",
    "gmail": "https://mail.google.com",
    "chatgpt": "https://chatgpt.com",
    "github": "https://github.com",
}


# 2. Open approved websites
def open_website(name):
    name = name.lower().strip()

    if name in WEBSITES:
        webbrowser.open(WEBSITES[name])
        return f"Opening {name}."

    return None


# 3. Search the web using your default browser
def search_web(query):
    query = query.strip()

    if not query:
        return "Tell me what you want to search for."

    url = "https://www.google.com/search?q=" + quote_plus(query)
    webbrowser.open(url)

    return f"Searching the web for {query}."


# 4. Open selected Windows applications
def open_application(name):
    name = name.lower().strip()

    applications = {
        "calculator": ["calc.exe"],
        "notepad": ["notepad.exe"],
        "paint": ["mspaint.exe"],
        "command prompt": ["cmd.exe"],
        "file explorer": ["explorer.exe"],
    }

    if name not in applications:
        return None

    try:
        subprocess.Popen(applications[name])
        return f"Opening {name}."

    except OSError as error:
        print("Application error:", error)
        return f"I couldn't open {name}."


# 5. Open common folders
def open_folder(name):
    folders = {
        "downloads": os.path.join(
            os.path.expanduser("~"), "Downloads"
        ),
        "documents": os.path.join(
            os.path.expanduser("~"), "Documents"
        ),
        "desktop": os.path.join(
            os.path.expanduser("~"), "Desktop"
        ),
    }

    name = name.lower().strip()

    if name not in folders:
        return None

    path = folders[name]

    if not os.path.isdir(path):
        return f"I couldn't find your {name} folder."

    try:
        os.startfile(path)
        return f"Opening your {name} folder."

    except OSError:
        return f"I couldn't open your {name} folder."


# 6. Process a computer-control command
def handle_computer_command(command):
    text = command.lower().strip()

    # Search the web
    search_prefixes = (
        "search the web for ",
        "search for ",
        "google ",
    )

    for prefix in search_prefixes:
        if text.startswith(prefix):
            query = command.strip()[len(prefix):].strip()
            return search_web(query)

    # Open a website
    if text.startswith("open "):
        target = text[5:].strip()

        # Check websites first
        response = open_website(target)
        if response:
            return response

        # Then applications
        response = open_application(target)
        if response:
            return response

        # Then folders
        response = open_folder(target)
        if response:
            return response

    return None


# 7. Run this file directly to test the module
if __name__ == "__main__":
    print("TURBO computer control test")
    print("Try: open youtube")
    print("Try: open calculator")
    print("Try: open downloads")
    print("Try: search for Python tutorials")
    print("Type exit to quit.")

    while True:
        command = input("\nCommand: ").strip()

        if command.lower() == "exit":
            break

        response = handle_computer_command(command)

        if response:
            print("TURBO:", response)
        else:
            print("TURBO: I don't recognize that computer command.")
