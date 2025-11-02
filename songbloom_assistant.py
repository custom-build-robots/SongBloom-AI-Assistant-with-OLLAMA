import gradio as gr
import requests
import json
import os
import subprocess
import shutil
import uuid
from typing import List, Tuple, Optional 
from pydub import AudioSegment 
from datetime import datetime # NEW: For Timestamp
import re # ADDED: For robust string parsing

# --- 1. Configuration (Config) ---
DEFAULT_OLLAMA_URL = "http://localhost:11434"
GRADIO_PORT = int(os.environ.get("GRADIO_PORT", 9012))
DEFAULT_OLLAMA_MODEL = "gpt-oss:20b"
TARGET_PROMPT_DURATION_MS = 10000 # 10 seconds in milliseconds

# --- PATH DEFINITIONS BASED ON INSTALLATION SCRIPT ---
# CRITICAL FIX: Removed the incorrect double-tilde (~~) from the path expansion.
SONGBLOOM_REPO_DIR = os.path.expanduser("~/SongBloom")
SONGBLOOM_INFERENCE_SCRIPT = os.path.join(SONGBLOOM_REPO_DIR, "infer.py")
SONGBLOOM_MODEL_DOWNLOAD_DIR = os.path.join(SONGBLOOM_REPO_DIR, "checkpoints")

# CRITICAL CHANGE: Archive folder inside the repository
TEMP_DIR = os.path.join(SONGBLOOM_REPO_DIR, "generated_songs_archive")


# The corrected LLM prompt that MUST generate the EXACT TOKEN format.
SONGBLOOM_SYSTEM_INSTRUCTION = """
You are an experienced AI songwriter and must generate lyrics EXACTLY according to the SongBloom token format.
Adhere STRICTLY to these rules and return ONLY the formatted lyrics, without introductory or concluding sentences.

Rules:
1. Vocal sections MUST begin with a structure flag ([verse], [chorus], [bridge]).
2. Non-vocal sections MUST be represented with a structure flag ([intro], [inst], [outro]).
3. Structure flags MUST be REPEATED according to the desired duration. Each token (e.g., [inst]) corresponds to 1 or 5 seconds, depending on the SongBloom model.
4. Sentences within a vocal section are separated by a period (".")
5. A comma (",") MUST ONLY be placed at the end of a vocal or non-vocal section to signal the transition.
6. The entire song should correspond to 2 to 4 minutes.

Example (EXACT expected output format):

[intro] [intro] [intro] [intro] [intro] [intro] [intro] , [verse] Rolling in the heat of a backroad dream. Whiskey nights and peach moonbeam. Porch talk slow like a lazy stream. Warm drawl and denim seams , [chorus] Barbecue smoke and backbeat soul. Stories lived where the wild winds roll. In every sway. A tale unfolds. Southern gold that never gets old. , [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] [inst] , [verse] Humming to the tune of an old love song. Summer breeze been blowing all night long. Echoes in the cypress still feel right. Even when we fight. we hold tight , [chorus] Barbecue smoke and backbeat soul. Stories lived where the wild winds roll. In every sway, a tale unfolds. Southern gold that never gets old. , [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro] [outro]
"""

# --- 2. Ollama Client Functions ---

def list_ollama_models(url: str) -> List[str]:
    """Fetches a list of available models from the Ollama server."""
    try:
        response = requests.get(f"{url}/api/tags", timeout=5)
        response.raise_for_status()
        models_data = response.json()
        
        models = [m['name'] for m in models_data.get('models', [])]
        
        # Sort models alphabetically
        models.sort()
        
        if not models:
            return ["ERROR: No models found on the Ollama server."]
        return models
    except requests.exceptions.ConnectionError:
        return [f"ERROR: Ollama server unreachable at {url}"]
    except requests.exceptions.RequestException as e:
        return [f"ERROR: Problem fetching models: {e}"]
    except json.JSONDecodeError:
        return [f"ERROR: Invalid JSON response from Ollama at {url}"]

def generate_lyrics_from_ollama(url: str, model_name: str, full_prompt: str) -> Tuple[str, str]:
    """Sends a prompt to Ollama to generate lyrics. Returns (raw_text, clean_text | error_message)"""
    payload = {
        "model": model_name,
        "prompt": full_prompt,
        "raw": True,
        "stream": False
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(f"{url}/api/generate", json=payload, headers=headers, timeout=120)
        response.raise_for_status()
        result = response.json()
        
        # Store the raw text including any preamble/thought process
        raw_text = result.get("response", "").strip() 
        generated_text = raw_text
        
        # --- REFINED FIX: Find the first token and strip internal monologue ---
        
        # 1. Strip the strongest conversational markers first, if present
        final_marker = '<|channel|>final<|message|>'
        if final_marker in generated_text:
             generated_text = generated_text.split(final_marker)[-1].strip()
        
        # Pattern to find the start of a SongBloom section token
        START_TOKENS_PATTERN = r'(\[intro\]|\[verse\]|\[chorus\]|\[bridge\])'
        start_match = re.search(START_TOKENS_PATTERN, generated_text, re.IGNORECASE)
        
        if start_match:
            # 2. Strip everything before the found token (removes planning/preamble)
            clean_text = generated_text[start_match.start():]
            
            # 3. Strip potential trailing LLM/model tags (e.g., <|end|>)
            clean_text = re.sub(r'(\<\|end\|\>|\<\|start\|\>assistant\<\|channel\|\>final\<\|message\|\>|\n|\r)', '', clean_text, flags=re.IGNORECASE).strip()
            
            if not clean_text:
                return raw_text, "ERROR: LLM output filtering resulted in an empty string."
            
            # 4. Format the cleaned text for display
            formatted_text = clean_text.replace(' ,', ' ,\n')
            
            return raw_text, formatted_text # Return both raw and clean
        else:
            # Filtering failed because the required token was not found
            return raw_text, "ERROR: LLM output filtering failed: Required SongBloom tokens were not found. Raw output provided."
        
    except requests.exceptions.RequestException as e:
        # API call failed
        return f"API ERROR: {e}", "" 
    except Exception as e:
        # Other exception
        return f"Unexpected error during Ollama call: {e}", ""

# --- 3. Prompt Engineering and Main Logic (LLM Text Generation) ---

def run_lyrics_generation(ollama_url: str, model_name: str, user_prompt_idea: str, target_length_minutes: int) -> Tuple[str, str, str]:
    """Executes the entire text generation process. Returns (raw_lyrics, clean_lyrics, status_log)"""
    if not model_name or "ERROR" in model_name:
        return "", "", f"ERROR: Please select a valid Ollama model. Current state: {model_name}"
    
    # Prompt combination
    full_prompt = (
        f"{SONGBLOOM_SYSTEM_INSTRUCTION}\n\n"
        f"Generate a song lyric in this exact token format. The song should correspond to approximately "
        f"{target_length_minutes} minutes in length.\n\n"
        f"User Idea:\n{user_prompt_idea}\n\n"
        f"**ABSOLUTELY CRITICAL: RETURN ONLY THE TOKENS AND LYRICS. DO NOT INCLUDE ANY PREAMBLE, COMMENTARY, OR EXPLANATION.**\n\n" 
        f"Lyrics:"
    )

    status_log = f"Attempting to generate lyrics with model '{model_name}'..."
    raw_lyrics, clean_lyrics = generate_lyrics_from_ollama(ollama_url, model_name, full_prompt)

    if raw_lyrics.startswith("API ERROR:"):
        # API failure: raw_lyrics holds error, clean_lyrics is ""
        return raw_lyrics, "", raw_lyrics
    elif clean_lyrics.startswith("ERROR:"):
        # Filtering failure: raw_lyrics holds raw, clean_lyrics holds error message
        return raw_lyrics, "", clean_lyrics
    else:
        status_log = f"Successfully generated with '{model_name}'. Duration: {target_length_minutes} minutes."
        return raw_lyrics, clean_lyrics, status_log

# --- NEW HELPER FUNCTION: Audio Preprocessing ---

def preprocess_audio_prompt(input_file_path: str) -> Tuple[Optional[str], str]:
    """
    Converts the style prompt file to the 10s WAV (48kHz) required by SongBloom.
    """
    unique_id = str(uuid.uuid4())
    # Temporary path for the processed WAV file (will be archived/not deleted)
    processed_wav_path = os.path.join(TEMP_DIR, f"prompt_runtime_{unique_id}.wav")

    try:
        # 1. Loading the audio file
        if input_file_path.lower().endswith('.mp3'):
            audio = AudioSegment.from_mp3(input_file_path)
            status_msg = "MP3 file recognized and loaded. "
        elif input_file_path.lower().endswith('.wav') or input_file_path.lower().endswith('.flac'):
            audio = AudioSegment.from_file(input_file_path)
            status_msg = "WAV/FLAC file loaded. "
        else:
            return None, f"ERROR: Unsupported file format. Please upload WAV, MP3, or FLAC."

        # 2. Length check and adjustment
        current_duration_ms = len(audio)
        
        if current_duration_ms > TARGET_PROMPT_DURATION_MS:
            audio = audio[:TARGET_PROMPT_DURATION_MS]
            status_msg += "File trimmed to 10 seconds. "
        elif current_duration_ms < TARGET_PROMPT_DURATION_MS:
            # Extending by repetition
            repeats = TARGET_PROMPT_DURATION_MS // current_duration_ms
            remainder_ms = TARGET_PROMPT_DURATION_MS % current_duration_ms
            
            filler = audio * repeats + audio[:remainder_ms]
            audio = filler
            status_msg += "File extended to 10 seconds by repetition. "
        else:
            status_msg += "Length matches (10 seconds). "

        # 3. Saving as WAV (48kHz, 1 channel)
        audio.export(processed_wav_path, format="wav", parameters=["-ac", "1", "-ar", "48000"]) 
        
        status_msg += f"Successfully saved as WAV (48kHz)."
        return processed_wav_path, status_msg

    except FileNotFoundError:
        return None, "ERROR: FFMPEG/LAME is not installed or in PATH. Please install FFMPEG."
    except Exception as e:
        return None, f"ERROR during audio preprocessing with pydub/ffmpeg: {e}"


# --- 4. Audio Generation Logic with infer.py ---

def generate_audio_from_lyrics(lyrics: str, prompt_wav_file: Optional[gr.File], songbloom_model_name: str, dtype: str, output_format: str) -> Tuple[Optional[str], str]:
    """
    Performs audio generation, archives intermediate results, and converts the output.
    """
    if not lyrics.strip():
        return None, "ERROR: No lyrics available for audio generation. Please generate lyrics first."
    
    if not prompt_wav_file or not os.path.exists(prompt_wav_file.name):
         return None, "ERROR: Please upload a 10s 'Style Prompt WAV/MP3' file."
    
    # Audio preprocessing (creates the temporary WAV file)
    processed_wav_path, pre_process_status = preprocess_audio_prompt(prompt_wav_file.name)
    if not processed_wav_path:
        return None, f"ERROR during audio preprocessing: {pre_process_status}"
    
    # NEW: Creation of base name with timestamp
    unique_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{timestamp}_{unique_id[:8]}" 
    
    # --- Define File Paths ---
    input_jsonl_path_runtime = os.path.join(TEMP_DIR, f"{base_name}_infer_input_runtime.jsonl") # Passed to infer.py at runtime
    
    lyrics_archive_path = os.path.join(TEMP_DIR, f"{base_name}_lyrics.txt") # Persistent archive path
    
    output_audio_dir_for_infer = TEMP_DIR # Main archive folder for infer.py output
    
    # NEW: Final file name based on timestamp
    final_audio_filename = f"{base_name}.{output_format.lower()}"
    expected_output_filename_flac = f"{unique_id}_s0.flac" # SongBloom output is always FLAC
    
    # JSONL Entry
    jsonl_entry = {
        "idx": unique_id,
        "lyrics": lyrics,
        "prompt_wav": processed_wav_path
    }
    
    # CRITICAL FIX: Securing the output_format
    if output_format is None or output_format not in ["FLAC", "WAV", "MP3"]:
         output_format = "WAV" # Fallback to default value
         
    
    try:
        # 1. Create JSONL file for infer.py (Runtime)
        with open(input_jsonl_path_runtime, 'w', encoding='utf-8') as f:
            f.write(json.dumps(jsonl_entry, ensure_ascii=False) + '\n')

        # 2. Archive the generated lyrics
        with open(lyrics_archive_path, 'w', encoding='utf-8') as f:
            f.write(lyrics)

        status_log = f"Audio preprocessing: {pre_process_status}\nStarting audio generation with '{songbloom_model_name}'..."

        if not os.path.exists(SONGBLOOM_INFERENCE_SCRIPT):
            return None, f"ERROR: 'infer.py' not found under {SONGBLOOM_INFERENCE_SCRIPT}. Check path."

        command = [
            "python3", SONGBLOOM_INFERENCE_SCRIPT,
            "--input-jsonl", input_jsonl_path_runtime, 
            "--output-dir", output_audio_dir_for_infer, 
            "--model-name", songbloom_model_name,
            "--local-dir", SONGBLOOM_MODEL_DOWNLOAD_DIR,
            "--dtype", dtype,
            "--n-samples", "1"
        ]
        
        env = os.environ.copy()
        process = subprocess.run(command, capture_output=True, text=True, env=env, cwd=SONGBLOOM_REPO_DIR, timeout=1800)

        if process.returncode != 0:
            return None, f"ERROR during audio generation (Exit code {process.returncode}):\nSTDOUT:\n{process.stdout}\nSTDERR:\n{process.stderr}"

        # **File Search & Archiving**
        original_flac_path = os.path.join(output_audio_dir_for_infer, expected_output_filename_flac)
        if not os.path.exists(original_flac_path):
             original_flac_path = os.path.join(output_audio_dir_for_infer, "0", expected_output_filename_flac)

        if os.path.exists(original_flac_path):
            
            final_audio_path = os.path.join(TEMP_DIR, final_audio_filename)
            
            # --- Conversion Logic ---
            if output_format.upper() == "FLAC":
                shutil.move(original_flac_path, final_audio_path)
                status_suffix = "(FLAC)."
            else: 
                try:
                    audio = AudioSegment.from_file(original_flac_path, format="flac")
                    audio.export(final_audio_path, format=output_format.lower()) 
                    os.remove(original_flac_path) # Delete original FLAC
                    status_suffix = f"and converted to {output_format}!"
                    
                except Exception as e:
                    fallback_path = os.path.join(TEMP_DIR, final_audio_filename.replace(output_format.lower(), "flac"))
                    shutil.move(original_flac_path, fallback_path)
                    return fallback_path, f"WARNING: Conversion to {output_format} failed ({e}). FLAC file loaded instead."
            
            # Archive the JSONL input file (rename and save)
            jsonl_archive_path = os.path.join(TEMP_DIR, f"{base_name}_infer_input.jsonl")
            # All intermediate and final files are now archived, nothing is deleted.
            shutil.move(input_jsonl_path_runtime, jsonl_archive_path)
            
            return final_audio_path, f"Audio successfully generated {status_suffix}"

        else:
            return None, f"ERROR: Generation successful according to log, but file '{expected_output_filename_flac}' not found. Archives remain empty."

    except Exception as e:
        stderr_log = process.stderr if 'process' in locals() else 'N/A'
        return None, f"An unexpected error occurred: {e}\nSTDERR:\n{stderr_log}"
    finally:
        # **FINALLY Block Cleanup:** REMOVED explicit deletion of processed_wav_path and JSONL file.
        # All files (preprocessed WAV, JSONL, final audio) are now persistent in TEMP_DIR.
        pass

# --- 5. Gradio Interface Design & Interactivity ---

with gr.Blocks(title="SongBloom Assistant") as app:
    gr.Markdown("# 🎵 SongBloom AI Assistant")
    gr.Markdown(
        f"Generate lyrics in **token format** via Ollama and convert them into **audio** using SongBloom. "
        f"All results are archived in `{TEMP_DIR}` and retain a timestamp."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Text Generation Configuration (Ollama)")
            ollama_url_input = gr.Textbox(label="Ollama Server URL", value=DEFAULT_OLLAMA_URL)
            
            ollama_model_dropdown = gr.Dropdown(label="Available Ollama Models", choices=[], interactive=True)
            refresh_ollama_models_btn = gr.Button("🔄 Update Ollama Models")
            target_length_slider = gr.Slider(minimum=1, maximum=15, value=2, step=1, label="Approximate Song Length (Minutes for LLM)")
            
            gr.Markdown("---")
            gr.Markdown("### 🔊 Audio Configuration (SongBloom)")
            
            prompt_wav_upload = gr.File(
                label="Style Prompt Audio (10s WAV/MP3) - Automatically converted/adjusted",
                file_types=[".wav", ".mp3", ".flac"], 
                interactive=True
            )
            
            # NEW RADIO BUTTON for the output format
            output_format_radio = gr.Radio(
                label="Output Format of the Generated Song",
                choices=["FLAC", "WAV", "MP3"],
                value="MP3", # CHANGED to MP3 as requested
                interactive=True
            )
            
            songbloom_model_name_input = gr.Textbox(
                label="SongBloom Audio Model Name (--model-name)", 
                value="songbloom_full_150s", 
                info="E.g. songbloom_full_150s"
            )
            songbloom_dtype_dropdown = gr.Dropdown(
                label="Data Type (--dtype)", 
                choices=["float32", "bfloat16", "float16"], 
                value="float32"
            )


        with gr.Column(scale=2):
            gr.Markdown("### 📝 Your Song Idea & Text Generation")
            user_prompt_input = gr.Textbox(label="Your Song Idea (Genre, Topic, Mood, Specifics)",
                placeholder="E.g.: 'An emotional pop song about lost love and new beginnings, melancholic mood.'", lines=5)
            generate_lyrics_button = gr.Button("🚀 Generate Lyrics", variant="primary")
            
            gr.Markdown("### 🎤 Generated Lyrics")
            
            # 1. FULL LLM OUTPUT (Read-only)
            raw_lyrics_output = gr.Textbox(
                label="1. Full LLM Output (Includes thought process/preamble)", 
                lines=10, 
                interactive=False, 
                show_copy_button=True
            )
            
            # 2. CLEAN LYRICS INPUT (Editable)
            clean_lyrics_input = gr.Textbox(
                label="2. Clean SongBloom Text (Editable Input for Audio Generation)", 
                lines=10, 
                interactive=True, 
                show_copy_button=True
            )
            
            # The old text_generation_status_log remains the same
            text_generation_status_log = gr.Textbox(label="Status / Log (Text Generation)", lines=2, interactive=False)
            
            gr.Markdown("---")
            generate_audio_button = gr.Button("▶️ Generate Audio", variant="secondary")

            gr.Markdown("### 🎧 Your Song (Audio)")
            audio_output = gr.Audio(label="Generated Audio", value=None, autoplay=False, interactive=False)
            audio_generation_status_log = gr.Textbox(label="Status / Log (Audio Generation)", lines=2, interactive=False)

    # --- Interactivity ---
    
    def update_ollama_model_dropdown(url: str) -> Tuple[gr.Dropdown, gr.Textbox]:
        models = list_ollama_models(url)
        
        # Set default value to DEFAULT_OLLAMA_MODEL or the first available model
        default_val = DEFAULT_OLLAMA_MODEL if DEFAULT_OLLAMA_MODEL in models else (models[0] if models and "ERROR" not in models[0] else None)
        
        if models and "ERROR" in models[0]:
            return gr.Dropdown(choices=models, value=models[0], interactive=False), gr.Textbox(value=models[0])
        else:
            return gr.Dropdown(choices=models, value=default_val, interactive=True), gr.Textbox(value="Ollama models loaded successfully.")

    app.load(update_ollama_model_dropdown, inputs=[ollama_url_input], outputs=[ollama_model_dropdown, text_generation_status_log])
    refresh_ollama_models_btn.click(update_ollama_model_dropdown, inputs=[ollama_url_input], outputs=[ollama_model_dropdown, text_generation_status_log])

    # Text Generation (Updates ALL three text fields now)
    generate_lyrics_button.click(
        run_lyrics_generation,
        inputs=[ollama_url_input, ollama_model_dropdown, user_prompt_input, target_length_slider],
        outputs=[raw_lyrics_output, clean_lyrics_input, text_generation_status_log] # NEW OUTPUTS
    )
    
    # Audio Generation (Reads from the new, editable clean_lyrics_input field)
    generate_audio_button.click(
        generate_audio_from_lyrics,
        inputs=[clean_lyrics_input, prompt_wav_upload, songbloom_model_name_input, songbloom_dtype_dropdown, output_format_radio], # UPDATED INPUT
        outputs=[audio_output, audio_generation_status_log]
    )
    
    # NEW FOOTER
    gr.Markdown(
        "<p style='text-align: center; color: #777; margin-top: 20px;'>Made with love by <a href='https://ai-box.eu/' target='_blank'>https://ai-box.eu/</a></p>"
    )

# --- App Start ---
if __name__ == "__main__":
    # Ensure the archive folder exists, but DO NOT delete it.
    os.makedirs(TEMP_DIR, exist_ok=True) 

    app.launch(server_name="0.0.0.0", server_port=GRADIO_PORT, share=False)
