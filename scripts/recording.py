#!/usr/bin/env python3
# recording.py - Piper GPL Training Recorder (Improved)
# Compatible with Python 3.13 (Windows 10/11, Linux Mint)
# Usage: python recording.py [--sentences FILENAME]

import os
import sys
import argparse
import tkinter as tk
from tkinter import messagebox, ttk
import sounddevice as sd
import soundfile as sf
import numpy as np
import threading

# -----------------------------
# Configuration
# -----------------------------
WAV_DIR = "my-training/wav"
METADATA_FILE = os.path.join(WAV_DIR, "metadata.csv")
DEFAULT_SENTENCES_FILE = "sentences.txt"
SAMPLE_RATE = 22050
LEVEL_THRESHOLD_LOW = 0.02
LEVEL_THRESHOLD_HIGH = 0.9
TARGET_RMS = 0.15  # Target RMS for normalization
SILENCE_THRESHOLD = 0.01  # Threshold for silence detection
SILENCE_DURATION = 0.1  # Seconds of silence to trim

# -----------------------------
# Default Sentences (35 optimized)
# -----------------------------
DEFAULT_SENTENCES = [
    # Contest and CQ calls
    "CQ Contest, CQ Contest, this is Whiskey Seven India Yankee.",
    "Alpha Bravo Three Charlie Delta calling CQ on twenty meters.",
    "Sierra Oscar Tango Alpha calling and listening.",
    
    # Signal reports and exchanges
    "Whiskey Seven India Yankee, you're five nine in zone five.",
    "You're five nine nine, name here is Mike, Mike.",
    "Your signal is five seven here in Virginia.",
    "November One Mike Mike, you're five nine, QSL?",
    "That's five nine nine, contest number two three four.",
    
    # Locations and technical details
    "My QTH is Grid Square Echo Mike Seven Three.",
    "Zone one four, state is California, over.",
    "Running one hundred watts to a dipole antenna.",
    
    # Procedural phrases
    "Roger, thanks for the contact, seven three.",
    "Kilo Four Zulu Echo Charlie, go ahead please.",
    "Frequency is clear, go ahead with your call.",
    "Good luck in the contest, Whiskey Seven India Yankee clear.",
    "Confirm your callsign is November Seven Bravo Romeo Charlie?",
    "Last two of serial number are eight seven.",
    
    # Questions and varied structures
    "What's your power output and antenna configuration?",
    "Can you hear me through the static and interference?",
    "Which band are you planning to operate on tonight?",
    "Have you worked any DX stations this morning?",
    "Are you using a vertical or horizontal polarization?",
    
    # Weather and conditions (phoneme diversity)
    "The weather here is cloudy with occasional showers.",
    "Propagation conditions are excellent on fifteen meters today.",
    "Heavy thunderstorms are affecting reception in the northeast.",
    
    # Equipment and setup
    "My transceiver is a modern digital radio with DSP.",
    "The antenna tuner matches impedance perfectly.",
    "I'm adjusting the microphone gain for better audio quality.",
    "Please switch to the upper sideband for this contact.",
    
    # Conversational and natural speech
    "I've been a licensed amateur radio operator for twelve years.",
    "The local club meets every Thursday evening at seven.",
    "Your audio sounds crisp and clear on my receiver.",
    "I enjoy chasing rare DX entities and collecting QSL cards.",
    "My favorite mode is CW, though I also enjoy phone contacts.",
    
    # Longer narrative sentences (prosody training)
    "During the contest, I managed to work stations in over forty countries across six continents."
]

# -----------------------------
# Load sentences from file
# -----------------------------
def load_sentences(filename):
    """Load sentences from a text file, one per line"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        if not sentences:
            print(f"Warning: No sentences found in {filename}, using defaults")
            return DEFAULT_SENTENCES
        print(f"Loaded {len(sentences)} sentences from {filename}")
        return sentences
    except FileNotFoundError:
        print(f"Sentences file '{filename}' not found, using defaults")
        return DEFAULT_SENTENCES
    except Exception as e:
        print(f"Error loading sentences file: {e}, using defaults")
        return DEFAULT_SENTENCES

def create_default_sentences_file(filename):
    """Create a default sentences.txt file if it doesn't exist"""
    if not os.path.exists(filename):
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for sentence in DEFAULT_SENTENCES:
                    f.write(sentence + '\n')
            print(f"Created default sentences file: {filename}")
        except Exception as e:
            print(f"Warning: Could not create sentences file: {e}")

# -----------------------------
# Audio Processing Functions
# -----------------------------
def normalize_audio(audio, target_rms=TARGET_RMS):
    """Normalize audio to target RMS level"""
    current_rms = np.sqrt(np.mean(audio ** 2))
    if current_rms > 0:
        scaling_factor = target_rms / current_rms
        # Prevent clipping
        audio = audio * scaling_factor
        max_val = np.max(np.abs(audio))
        if max_val > 0.95:
            audio = audio * (0.95 / max_val)
    return audio

def trim_silence(audio, sample_rate, threshold=SILENCE_THRESHOLD, duration=SILENCE_DURATION):
    """Trim silence from beginning and end of audio"""
    # Calculate frames for minimum duration
    min_frames = int(duration * sample_rate)
    
    # Find non-silent regions
    abs_audio = np.abs(audio)
    
    # Smooth with a moving average to avoid cutting on brief dips
    window = min(512, len(abs_audio) // 10)
    if window > 1:
        smoothed = np.convolve(abs_audio, np.ones(window)/window, mode='same')
    else:
        smoothed = abs_audio
    
    # Find start
    start_idx = 0
    for i in range(len(smoothed)):
        if smoothed[i] > threshold:
            start_idx = max(0, i - int(0.05 * sample_rate))  # Keep 50ms before speech
            break
    
    # Find end
    end_idx = len(smoothed)
    for i in range(len(smoothed) - 1, -1, -1):
        if smoothed[i] > threshold:
            end_idx = min(len(smoothed), i + int(0.05 * sample_rate))  # Keep 50ms after speech
            break
    
    # Ensure we don't trim too much
    if end_idx - start_idx < min_frames:
        return audio
    
    return audio[start_idx:end_idx]

# -----------------------------
# Recorder App
# -----------------------------
class RecorderApp:
    def __init__(self, master, sentences):
        self.master = master
        self.sentences = sentences
        self.index = 0
        self.recording = False
        self.audio_data = []
        self.stream = None

        # Ensure wav directory exists
        os.makedirs(WAV_DIR, exist_ok=True)

        # Load existing metadata if any
        self.load_metadata()

        # Build GUI
        self.build_gui()
        
        # Skip to first unrecorded sentence
        self.skip_to_next_unrecorded()
        self.update_sentence_display()

    # -----------------------------
    # GUI Setup
    # -----------------------------
    def build_gui(self):
        self.master.title("Piper GPL Recorder")
        self.master.geometry("700x400")
        
        # Progress info
        progress_frame = tk.Frame(self.master)
        progress_frame.pack(pady=10)
        
        self.progress_var = tk.StringVar()
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress_var, font=("Arial", 12))
        self.progress_label.pack()
        
        # Progress bar
        self.progress_bar = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.progress_bar.pack(pady=5)
        
        # Sentence display
        sentence_frame = tk.Frame(self.master, bg="lightgray", relief=tk.SUNKEN, bd=2)
        sentence_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        self.sentence_label = tk.Label(sentence_frame, text="", font=("Arial", 16, "bold"), 
                                       wraplength=650, bg="lightgray", justify=tk.CENTER)
        self.sentence_label.pack(pady=20, padx=10, expand=True)

        # Status label
        self.status_var = tk.StringVar(value="Ready to record")
        self.status_label = tk.Label(self.master, textvariable=self.status_var, 
                                     font=("Arial", 10), fg="blue")
        self.status_label.pack(pady=5)

        # Level meter - Single LED indicator
        meter_frame = tk.Frame(self.master)
        meter_frame.pack(pady=5)
        tk.Label(meter_frame, text="Level:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Create single large LED indicator
        self.led_canvas = tk.Canvas(meter_frame, width=50, height=50, bg="gray20", 
                                     highlightthickness=2, highlightbackground="gray40")
        self.led_canvas.pack(side=tk.LEFT, padx=10)
        self.led_circle = self.led_canvas.create_oval(8, 8, 42, 42, fill="gray30", outline="gray50", width=2)
        
        # Status label next to LED
        self.level_status_var = tk.StringVar(value="Ready")
        self.level_status_label = tk.Label(meter_frame, textvariable=self.level_status_var, 
                                           font=("Arial", 10), width=12)
        self.level_status_label.pack(side=tk.LEFT, padx=5)

        # Buttons
        button_frame = tk.Frame(self.master)
        button_frame.pack(pady=15)
        
        self.start_btn = tk.Button(button_frame, text="Start/Stop (Space)", 
                                   command=self.toggle_recording, width=15, height=2)
        self.start_btn.grid(row=0, column=0, padx=5)
        
        self.skip_btn = tk.Button(button_frame, text="Skip", 
                                  command=self.skip_sentence, width=12, height=2)
        self.skip_btn.grid(row=0, column=1, padx=5)
        
        self.review_btn = tk.Button(button_frame, text="Review", 
                                    command=self.review, width=12, height=2)
        self.review_btn.grid(row=0, column=2, padx=5)
        
        self.quit_btn = tk.Button(button_frame, text="Quit", 
                                  command=self.master.quit, width=12, height=2)
        self.quit_btn.grid(row=0, column=3, padx=5)

        # Key bindings
        self.master.bind("<space>", lambda e: self.toggle_recording())
        self.master.bind("<Right>", lambda e: self.skip_sentence())
        self.master.bind("<Left>", lambda e: self.previous_sentence())

    # -----------------------------
    # Sentence Management
    # -----------------------------
    def update_sentence_display(self):
        recorded_count = sum(1 for i in range(len(self.sentences)) if self.is_recorded(i))
        
        if self.index < len(self.sentences):
            self.sentence_label.config(text=self.sentences[self.index])
            status = "✓ Already recorded" if self.is_recorded(self.index) else "Press Space to record"
            self.progress_var.set(f"Sentence {self.index + 1} of {len(self.sentences)} "
                                 f"({recorded_count} recorded)")
        else:
            self.sentence_label.config(text="All sentences recorded!")
            self.progress_var.set(f"Complete! {recorded_count}/{len(self.sentences)} recorded")
            status = "All done!"
        
        # Update progress bar
        self.progress_bar['maximum'] = len(self.sentences)
        self.progress_bar['value'] = recorded_count
        
        if not self.recording:
            self.status_var.set(status)

    def skip_to_next_unrecorded(self):
        """Skip to the first unrecorded sentence"""
        while self.index < len(self.sentences) and self.is_recorded(self.index):
            self.index += 1

    def next_sentence(self):
        self.index += 1
        self.skip_to_next_unrecorded()
        self.update_sentence_display()

    def previous_sentence(self):
        """Go back to previous sentence"""
        if self.index > 0:
            self.index -= 1
            self.update_sentence_display()

    def skip_sentence(self):
        self.next_sentence()

    def is_recorded(self, idx):
        filename = f"{idx + 1:03d}.wav"
        return filename in [entry[0] for entry in self.metadata]

    # -----------------------------
    # Recording
    # -----------------------------
    def toggle_recording(self):
        if self.recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self):
        if self.index >= len(self.sentences):
            messagebox.showinfo("Complete", "All sentences have been recorded!")
            return
        
        self.recording = True
        self.audio_data = []
        self.status_var.set("🔴 RECORDING... (Press Space to stop)")
        self.sentence_label.config(bg="lightcoral")
        self.start_btn.config(relief=tk.SUNKEN)
        
        try:
            self.stream = sd.InputStream(channels=1, samplerate=SAMPLE_RATE, 
                                        callback=self.audio_callback)
            self.stream.start()
            self.update_meter()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start recording: {e}")
            self.recording = False
            self.status_var.set("Error starting recording")

    def stop_recording(self):
        self.recording = False
        self.sentence_label.config(bg="lightgray")
        self.start_btn.config(relief=tk.RAISED)
        self.status_var.set("Processing audio...")
        
        if self.stream:
            self.stream.stop()
            self.stream.close()
        
        if not self.audio_data:
            self.status_var.set("No audio recorded!")
            self.update_sentence_display()
            return
        
        # Convert to numpy array
        audio = np.concatenate(self.audio_data, axis=0).flatten()
        
        # Check RMS level
        rms = np.sqrt(np.mean(audio ** 2))
        
        if rms < LEVEL_THRESHOLD_LOW:
            messagebox.showwarning("Warning", 
                f"Audio level too low (RMS: {rms:.3f})!\nPlease speak louder and re-record.")
            self.status_var.set("Recording failed - too quiet")
            self.update_sentence_display()
        elif rms > LEVEL_THRESHOLD_HIGH:
            messagebox.showwarning("Warning", 
                f"Audio level too high (RMS: {rms:.3f})!\nPlease reduce volume and re-record.")
            self.status_var.set("Recording failed - too loud")
            self.update_sentence_display()
        else:
            try:
                # Trim silence
                audio = trim_silence(audio, SAMPLE_RATE)
                
                # Normalize
                audio = normalize_audio(audio)
                
                # Save WAV with 3-digit numbering
                filename = f"{self.index + 1:03d}.wav"
                filepath = os.path.join(WAV_DIR, filename)
                sf.write(filepath, audio, SAMPLE_RATE)
                
                # Save metadata
                self.save_metadata(filename, self.sentences[self.index])
                
                self.status_var.set(f"✓ Saved {filename} (RMS: {rms:.3f})")
                self.next_sentence()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save audio: {e}")
                self.status_var.set("Error saving audio")
                self.update_sentence_display()
        
        # Reset LED
        self.led_canvas.itemconfig(self.led_circle, fill="gray30", outline="gray50")
        self.level_status_var.set("Ready")

    def audio_callback(self, indata, frames, time, status):
        if self.recording:
            self.audio_data.append(indata.copy())

    def update_meter(self):
        if self.recording:
            if self.audio_data:
                audio = np.concatenate(self.audio_data, axis=0)
                # Use peak level instead of RMS for more responsive meter
                level = np.max(np.abs(audio[-SAMPLE_RATE//10:]))  # Last 0.1 seconds
                
                # Update single LED with color based on level
                if level < 0.05:
                    # Too quiet
                    color = "gray40"
                    outline = "gray50"
                    status = "Too quiet"
                elif level < 0.7:
                    # Good level - green
                    color = "lime"
                    outline = "green"
                    status = "Good"
                elif level < 0.9:
                    # Getting loud - yellow
                    color = "yellow"
                    outline = "orange"
                    status = "Loud"
                else:
                    # Too loud/clipping - red
                    color = "red"
                    outline = "darkred"
                    status = "Clipping!"
                
                self.led_canvas.itemconfig(self.led_circle, fill=color, outline=outline)
                self.level_status_var.set(status)
            self.master.after(50, self.update_meter)

    # -----------------------------
    # Metadata
    # -----------------------------
    def save_metadata(self, filename, sentence):
        # Update metadata list
        self.metadata.append([filename, sentence])
        
        # Rewrite entire metadata file sorted by filename
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            for entry in sorted(self.metadata, key=lambda x: x[0]):
                f.write(f"{entry[0]}|{entry[1]}\n")

    def load_metadata(self):
        self.metadata = []
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        self.metadata.append([parts[0], parts[1]])

    # -----------------------------
    # Review
    # -----------------------------
    def review(self):
        ReviewWindow(self.master, self)

# -----------------------------
# Review Window
# -----------------------------
class ReviewWindow:
    def __init__(self, master, app):
        self.top = tk.Toplevel(master)
        self.top.title("Review Recordings")
        self.top.geometry("700x350")
        self.app = app
        self.metadata = app.metadata.copy()
        self.index = 0

        # Sentence label
        self.sentence_label = tk.Label(self.top, text="", font=("Arial", 14), 
                                      wraplength=650, justify=tk.CENTER)
        self.sentence_label.pack(pady=15)

        # Progress
        self.progress_var = tk.StringVar()
        tk.Label(self.top, textvariable=self.progress_var).pack()

        # Level info
        self.level_label = tk.Label(self.top, text="", font=("Arial", 10))
        self.level_label.pack(pady=10)
        
        # Waveform visualization placeholder
        self.wave_canvas = tk.Canvas(self.top, width=600, height=80, bg="black")
        self.wave_canvas.pack(pady=10)

        # Buttons
        btn_frame = tk.Frame(self.top)
        btn_frame.pack(pady=15)
        
        self.prev_btn = tk.Button(btn_frame, text="← Previous", command=self.previous_sentence, width=12)
        self.prev_btn.grid(row=0, column=0, padx=5)
        
        self.play_btn = tk.Button(btn_frame, text="▶ Play", command=self.play_audio, width=12)
        self.play_btn.grid(row=0, column=1, padx=5)
        
        self.keep_btn = tk.Button(btn_frame, text="Keep →", command=self.next_sentence, width=12)
        self.keep_btn.grid(row=0, column=2, padx=5)
        
        self.delete_btn = tk.Button(btn_frame, text="🗑 Delete", command=self.delete_audio, 
                                    width=12, fg="red")
        self.delete_btn.grid(row=0, column=3, padx=5)
        
        self.quit_btn = tk.Button(btn_frame, text="Close", command=self.close_window, width=12)
        self.quit_btn.grid(row=0, column=4, padx=5)

        # Key bindings
        self.top.bind("<space>", lambda e: self.play_audio())
        self.top.bind("<Right>", lambda e: self.next_sentence())
        self.top.bind("<Left>", lambda e: self.previous_sentence())
        self.top.bind("<Delete>", lambda e: self.delete_audio())

        self.update_display()

    def update_display(self):
        if not self.metadata:
            self.sentence_label.config(text="No recordings to review")
            self.level_label.config(text="")
            self.progress_var.set("0/0")
            return
            
        if self.index < len(self.metadata):
            filename, sentence = self.metadata[self.index]
            self.sentence_label.config(text=sentence)
            self.progress_var.set(f"{self.index + 1}/{len(self.metadata)}")
            
            # Compute level info and draw waveform
            filepath = os.path.join(WAV_DIR, filename)
            if os.path.exists(filepath):
                try:
                    data, sr = sf.read(filepath)
                    min_lvl = np.min(data)
                    max_lvl = np.max(data)
                    rms_lvl = np.sqrt(np.mean(data**2))
                    duration = len(data) / sr
                    self.level_label.config(
                        text=f"Duration: {duration:.2f}s | Min: {min_lvl:.3f} | Max: {max_lvl:.3f} | RMS: {rms_lvl:.3f}"
                    )
                    self.draw_waveform(data)
                except Exception as e:
                    self.level_label.config(text=f"Error reading file: {e}")
            else:
                self.level_label.config(text="File missing!")
        else:
            self.sentence_label.config(text="End of recordings")
            self.level_label.config(text="")
            self.wave_canvas.delete("all")

    def draw_waveform(self, data):
        """Simple waveform visualization"""
        self.wave_canvas.delete("all")
        width = 600
        height = 80
        
        # Downsample for display
        step = max(1, len(data) // width)
        display_data = data[::step]
        
        # Normalize to canvas height
        if len(display_data) > 0:
            max_val = max(np.max(np.abs(display_data)), 0.01)
            display_data = display_data / max_val * (height / 2 - 5)
        
        # Draw waveform
        center = height / 2
        for i, val in enumerate(display_data):
            x = i * width / len(display_data)
            self.wave_canvas.create_line(x, center, x, center - val, fill="lime", width=1)

    def play_audio(self):
        if self.index < len(self.metadata):
            filename, _ = self.metadata[self.index]
            filepath = os.path.join(WAV_DIR, filename)
            if os.path.exists(filepath):
                try:
                    data, sr = sf.read(filepath)
                    sd.play(data, sr)
                    sd.wait()
                except Exception as e:
                    messagebox.showerror("Error", f"Failed to play audio: {e}")

    def next_sentence(self):
        if self.index < len(self.metadata) - 1:
            self.index += 1
            self.update_display()

    def previous_sentence(self):
        if self.index > 0:
            self.index -= 1
            self.update_display()

    def delete_audio(self):
        if self.index < len(self.metadata):
            filename, sentence = self.metadata[self.index]
            
            result = messagebox.askyesno("Confirm Delete", 
                f"Delete recording:\n\n{sentence}\n\nThis cannot be undone!")
            
            if result:
                filepath = os.path.join(WAV_DIR, filename)
                if os.path.exists(filepath):
                    try:
                        os.remove(filepath)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to delete file: {e}")
                        return
                
                # Remove from metadata
                self.metadata.pop(self.index)
                self.app.metadata = self.metadata.copy()
                
                # Rewrite metadata file
                with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                    for entry in sorted(self.metadata, key=lambda x: x[0]):
                        f.write(f"{entry[0]}|{entry[1]}\n")
                
                # Stay at same index (which now shows next item)
                if self.index >= len(self.metadata) and self.index > 0:
                    self.index -= 1
                
                self.update_display()
                
                # Update main window
                self.app.update_sentence_display()

    def close_window(self):
        self.app.update_sentence_display()
        self.top.destroy()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Piper GPL Training Recorder')
    parser.add_argument('--sentences', type=str, default=DEFAULT_SENTENCES_FILE,
                       help=f'Path to sentences file (default: {DEFAULT_SENTENCES_FILE})')
    args = parser.parse_args()
    
    # Create default sentences file if it doesn't exist
    create_default_sentences_file(DEFAULT_SENTENCES_FILE)
    
    # Load sentences from file
    sentences = load_sentences(args.sentences)
    
    print(f"Starting Piper GPL Recorder with {len(sentences)} sentences")
    print(f"Sentences file: {args.sentences}")
    print(f"Output directory: {WAV_DIR}/")
    print(f"Metadata file: {METADATA_FILE}")
    
    # Start GUI
    root = tk.Tk()
    app = RecorderApp(root, sentences)
    root.mainloop()

