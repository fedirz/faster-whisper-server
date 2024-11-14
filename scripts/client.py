import os
from pathlib import Path
import subprocess
import threading
import time
import httpx
import keyboard

# This script will run in the background and listen for a keybind to start recording audio.
# It will then wait until the keybind is pressed again to stop recording.
# The audio file will be sent to the server for transcription, and the transcription
# will be copied to the clipboard.

CHUNK = 2**12
AUDIO_RECORD_CMD = [
    "ffmpeg",
    "-hide_banner",
    "-f",
    "alsa",
    "-i",
    "default",
    "-f",
    "wav",
]
COPY_TO_CLIPBOARD_CMD = "wl-copy"
OPENAI_BASE_URL = "ws://localhost:8000/v1"
TRANSCRIBE_PATH = "/audio/transcriptions?language=en"
USER = "nixos"
TIMEOUT = httpx.Timeout(None)
KEYBIND = "ctrl+x"
REQUEST_KWARGS = {
    "language": "en",
    "response_format": "text",
    "vad_filter": True,
}

client = httpx.Client(base_url=OPENAI_BASE_URL, timeout=TIMEOUT)
is_running = threading.Event()

file = Path("test.wav")

def record_audio():
    """Records audio until the keybind is pressed again."""
    try:
        print("Recording started")
        process = subprocess.Popen(
            [*AUDIO_RECORD_CMD, "-y", str(file.name)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            user=USER,
            env=dict(os.environ),
        )
        return process
    except Exception as e:
        print(f"Error starting audio recording: {e}")
        return None

def stop_audio(process):
    """Stops the audio recording process."""
    if process:
        process.kill()
        stdout, stderr = process.communicate()
        if stdout or stderr:
            print(f"stdout: {stdout}")
            print(f"stderr: {stderr}")
        print(f"Recording finished. File size: {file.stat().st_size} bytes")

def transcribe_audio():
    """Sends the recorded audio file for transcription."""
    try:
        with file.open("rb") as fd:
            start = time.perf_counter()
            res = client.post(
                OPENAI_BASE_URL + TRANSCRIBE_PATH,
                files={"file": fd},
                data=REQUEST_KWARGS,
            )
            end = time.perf_counter()
            print(f"Transcription took {end - start} seconds")
            return res.text
    except httpx.ConnectError as e:
        print(f"Couldn't connect to server: {e}")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def copy_to_clipboard(transcription):
    """Copies the transcription to the clipboard."""
    try:
        subprocess.run([COPY_TO_CLIPBOARD_CMD], input=transcription.encode(), check=True)
        print("Transcription copied to clipboard.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to copy transcription to clipboard: {e}")
    except Exception as e:
        print(f"Error copying to clipboard: {e}")

def main():
    """Main loop for handling keybinds and recording/transcribing audio."""
    while True:
        try:
            # Wait for the keybind to start recording
            print(f"Press {KEYBIND} to start recording...")
            keyboard.wait(KEYBIND)
            process = record_audio()

            # Wait for the keybind to stop recording
            print(f"Press {KEYBIND} to stop recording...")
            keyboard.wait(KEYBIND)
            stop_audio(process)

            # Transcribe the recorded audio
            transcription = transcribe_audio()
            if transcription:
                print(f"Transcription: {transcription}")
                copy_to_clipboard(transcription)

        except (KeyboardInterrupt, SystemExit):
            print("Exiting...")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
