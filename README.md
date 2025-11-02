# 🎵 SongBloom AI Assistant (Ollama & Gradio UI)

## Overview

The SongBloom AI Assistant is a powerful local application designed to bridge the gap between creative songwriting and the complex technical requirements of modern AI music generation. It combines a local Large Language Model (LLM) running via Ollama to generate structured lyrics with the audio synthesis power of the Tencent SongBloom framework.

This assistant simplifies the process of creating full-length, coherent songs by ensuring that user-provided lyrics are generated in the exact Token Format required by SongBloom's inference script.

A powerful local AI assistant built with Python and Gradio that simplifies the creation of full-length AI songs. It generates highly specific, token-formatted lyrics (using local LLMs via Ollama) and pipes them directly to the Tencent SongBloom inference script to produce audio. Features include automatic audio format conversion (MP3/FLAC to WAV and vice-versa) and full asset archiving with timestamps.

## ✨ Features

*   **Token-Format Generation:** Uses Ollama to generate lyrics strictly adhering to the complex `[verse] [chorus]` SongBloom format.
*   **Editable Output:** Provides a clean, editable text field for manual adjustments to the tokenized lyrics before audio generation.
*   **Automatic Preprocessing:** Uploaded style prompt audio (MP3, FLAC, WAV) is automatically converted to the required 10-second, 48kHz WAV format.
*   **Flexible Output:** Choose your final song output format: FLAC, WAV, or MP3.
*   **Asset Archiving:** All generated assets (final song, clean lyrics, raw LLM output, inference JSONL, and preprocessed prompt audio) are permanently saved with a timestamp in the `generated_songs_archive/` folder for full traceability.
*   **Installation Script:** Includes a robust `install_songbloom_web.sh` script for easy setup on Ubuntu systems.

## 🛠️ Installation

### Prerequisites

You must have Ollama running on your system/network and have the necessary SongBloom models downloaded locally.

On Ubuntu, you must install `ffmpeg` for the audio conversion features to work:
```
sudo apt update && sudo apt install -y ffmpeg
```
### Setup using the Installation Script

Use the provided `install_songbloom_web.sh` script to set up the repository and a dedicated Python virtual environment:

Clone this repository (assuming you have already cloned the official SongBloom repo)
Navigate to where you saved the install script and run it:
    ```
./install_songbloom_web.sh
If successful, activate the new environment:
source ~/SongBloom/venv_songbloom/bin/activate
    ```

## 🚀 Usage

1.  **Start the Web UI:**

    ```
    python3 write_me_a_song.py
    ```

2.  **Configure LLM:** Open the browser interface (default: `http://localhost:9012`), verify the Ollama URL, and select your preferred LLM (e.g., `gpt-oss:20b`).

3.  **Generate Lyrics:** Enter your creative idea (Genre, Mood, Topic) and click "🚀 Generate Lyrics".

    The "Clean SongBloom Text" field will show the ready-to-use, tokenized lyrics. Edit this field if necessary.

4.  **Generate Audio:**

    *   Upload your 10s Style Prompt Audio.
    *   Select your desired Output Format (MP3 recommended for size).
    *   Click "▶️ Generate Audio".

    The application will pipe the clean lyrics to `infer.py` and output the final song directly in the audio player.

## 💾 Generated Files & Archiving

All final and intermediate files are archived under your main SongBloom directory:

`~/SongBloom/generated_songs_archive/`

Files are named using a timestamp and a unique ID for easy chronological lookup:

*   `YYYYMMDD_HHMMSS_ID_lyrics.txt` (Clean lyrics used)
*   `YYYYMMDD_HHMMSS_ID_infer_input.jsonl` (Input file used by `infer.py`)
*   `YYYYMMDD_HHMMSS_ID.mp3` (Final generated song)
*   `prompt_runtime_UUID.wav` (Preprocessed 10s prompt audio)
