#!/bin/bash
# Author: Gemini
# Date: 2025-11-01
# Version: 0.2
# The script installs SongBloom and the necessary packages for the Gradio Web App 
# (Gradio, Requests) into a Python 3.8 virtual environment.

# Exit immediately if a command exits with a non-zero status
set -e

# --- Konfiguration ---
PYTHON_VERSION="3.8"
REPO_URL="https://github.com/tencent-ailab/SongBloom.git"
REPO_DIR="$HOME/SongBloom"
VENV_NAME="venv_songbloom"
# ---------------------

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "--- Starte SongBloom Installations-Setup ---"

## 1. Systemvoraussetzungen prüfen und installieren ##

# 1.1 Python 3.8 und venv prüfen/installieren
if ! command_exists python${PYTHON_VERSION}; then
    echo "Python ${PYTHON_VERSION} ist nicht installiert. Versuche Installation..."
    sudo apt update
    # Installiere Python 3.8, venv und git
    sudo apt install -y python${PYTHON_VERSION} python${PYTHON_VERSION}-venv python${PYTHON_VERSION}-dev git
else
    echo "Python ${PYTHON_VERSION} ist installiert."
fi

# 1.2 libsndfile prüfen/installieren (erforderlich für Audioverarbeitung)
if ! dpkg -s libsndfile1 >/dev/null 2>&1; then
    echo "libsndfile ist nicht installiert. Installiere libsndfile..."
    # Update zuerst, um Paketlisten zu aktualisieren
    sudo apt update
    sudo apt install -y libsndfile1
else
    echo "libsndfile ist bereits installiert."
fi

# 1.3 Systempakete aktualisieren
echo "Aktualisiere die restlichen Systempakete..."
sudo apt update && sudo apt upgrade -y

# ---------------------------------------------------

## 2. Repository klonen und vorbereiten ##

# Klone das SongBloom Repository, falls es nicht existiert
if [ ! -d "$REPO_DIR" ]; then
    echo "Klone SongBloom Repository von $REPO_URL..."
    git clone "$REPO_URL" "$REPO_DIR"
else
    echo "SongBloom Repository existiert bereits. Lade neueste Änderungen..."
    cd "$REPO_DIR"
    git pull
fi

cd "$REPO_DIR"

# ---------------------------------------------------

## 3. Virtuelle Umgebung erstellen und Abhängigkeiten installieren ##

# Erstelle die virtuelle Umgebung, falls sie nicht existiert
if [ ! -d "$VENV_NAME" ]; then
    echo "Erstelle eine Python virtuelle Umgebung ($VENV_NAME) mit python${PYTHON_VERSION}..."
    python${PYTHON_VERSION} -m venv "$VENV_NAME"
else
    echo "Virtuelle Umgebung ($VENV_NAME) existiert bereits."
fi

# Aktiviere die virtuelle Umgebung
echo "Aktiviere die virtuelle Umgebung..."
source "$VENV_NAME/bin/activate"

# Upgrade pip
echo "Upgrade pip..."
pip install --upgrade pip

# 3.1 Installiere SongBloom Kern-Abhängigkeiten
echo "Installiere SongBloom Python-Abhängigkeiten (einschließlich torch 2.2.0 für CUDA 11.8)..."
# Der torch/torchaudio-Teil wird über requirements.txt installiert, was die empfohlene Version sicherstellt.
pip install -r requirements.txt

# 3.2 Installiere Web- und API-Tools (Gradio, Requests)
echo "Installiere Gradio und Requests für die Web-App..."
pip install gradio requests
pip install typer==0.9.4 --no-deps

# Deaktiviere die virtuelle Umgebung
echo "Deaktiviere die virtuelle Umgebung..."
deactivate

# ---------------------------------------------------

## ✅ Installation abgeschlossen und Nächste Schritte

echo "SongBloom und die Web-App-Voraussetzungen (Gradio, Requests) wurden erfolgreich installiert."
echo ""
echo "🔥 Nächste Schritte:"
echo "1. Wechsle in das Verzeichnis:"
echo "   cd $REPO_DIR"
echo "2. Erstelle die Web-Anwendung 'write_me_a_song.py' (Code wurde zuvor bereitgestellt)."
echo "3. Aktiviere die Umgebung:"
echo "   source $VENV_NAME/bin/activate"
echo "4. Starte die Gradio App:"
echo "   python3 write_me_a_song.py"