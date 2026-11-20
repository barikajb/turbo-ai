
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# Allow TURBO to access files created or opened by this app
SCOPES = ["https://www.googleapis.com/auth/drive.file"]


def main():
    credentials = None

    # Reuse a saved login if one exists
    if os.path.exists("token.json"):
        credentials = Credentials.from_authorized_user_file(
            "token.json", SCOPES
        )

    # Sign in if necessary
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            credentials = flow.run_local_server(port=0)

        # Save login details for future runs
        with open("token.json", "w", encoding="utf-8") as token:
            token.write(credentials.to_json())

    # Connect to Google Drive
    service = build("drive", "v3", credentials=credentials)

    # Test the connection
    results = service.files().list(
        pageSize=10,
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get("files", [])

    print("\nTURBO successfully connected to Google Drive!")
    print("Connection test complete.")

    if files:
        print("\nFiles visible to TURBO:")
        for file in files:
            print(f"- {file['name']} ({file['mimeType']})")
    else:
        print("\nNo files are visible to TURBO yet. That's okay.")


if __name__ == "__main__":
    main()
