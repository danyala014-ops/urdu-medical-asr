"""Live microphone transcription (Urdu + English) with any model from asr_engine.py.

Usage:
    python live_transcribe.py                  # pick a microphone from a list, Ctrl+C to stop
    python live_transcribe.py --model whisper  # Whisper turbo (+ cardiology prompt; no live preview)
    python live_transcribe.py --model parakeet # NVIDIA Parakeet TDT 0.6B int8, CPU-optimized
    python live_transcribe.py --model omni     # Meta Omnilingual 300M: Urdu (and English), CPU
    python live_transcribe.py --model omni-urdu  # ...forced to write Urdu script
    python live_transcribe.py --model omni-ft    # our Urdu fine-tune (models/omni-urdu-ft/)
    python live_transcribe.py --device 2       # use mic #2 from the list (or a name, e.g. headset)
    python live_transcribe.py --list-devices   # show available microphones and exit
    python live_transcribe.py --file clip.wav  # simulate live input from a file (real-time pace)
    python live_transcribe.py --no-lm          # skip the cardiology LM

How it works: Silero VAD splits the audio stream into utterances. While you speak,
the current utterance is re-transcribed every PARTIAL_EVERY_S seconds and shown as a
live preview line (fast greedy decoding); after END_SILENCE_MS of silence it is finalized
with the cardiology LM (lm/cath_lm.pkl.gz, if built) and saved.
"""

import os
import queue
import shutil
import sys
import threading
import time
import warnings
from collections import deque
from datetime import datetime
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

print("starting up (loading libraries, ~10-20 s)...", flush=True)

import librosa
import numpy as np
import soundfile as sf
import torch
from silero_vad import load_silero_vad

from asr_engine import DEFAULT_LM_PATH, SAMPLE_RATE, load_asr
from microphones import choose_device, list_input_devices, print_devices

FRAME = 512                  # Silero VAD frame size at 16 kHz (32 ms)
SPEECH_PROB = 0.5            # VAD prob to start an utterance
SILENCE_PROB = 0.35          # VAD prob below which a frame counts as silence
END_SILENCE_MS = 1000        # pause length that finalizes an utterance
PRE_ROLL_MS = 500            # audio kept from before speech onset
PARTIAL_EVERY_S = 0.6        # live preview refresh interval
MAX_UTTERANCE_S = 15         # force-finalize long utterances to bound latency...
SPLIT_SEARCH_S = 3           # ...cutting at the quietest point within this many final seconds
TRANSCRIPTS_DIR = Path(__file__).parent / "transcripts"

FRAME_MS = FRAME * 1000 / SAMPLE_RATE


def mic_frames(device, stop):
    import sounddevice as sd

    q = queue.Queue()
    with sd.InputStream(device=device, samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                        blocksize=FRAME,
                        callback=lambda data, *_: q.put(data[:, 0].copy())):
        while not stop.is_set():
            try:
                yield q.get(timeout=0.1)  # timeout keeps Ctrl+C responsive on Windows
            except queue.Empty:
                continue


def file_frames(path, stop):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    audio = np.concatenate([audio, np.zeros(SAMPLE_RATE, dtype=np.float32)])  # trailing silence
    q = queue.Queue()

    def feed():  # push frames at real-time pace, like a microphone would
        for i in range(0, len(audio) - FRAME + 1, FRAME):
            q.put(audio[i:i + FRAME])
            time.sleep(FRAME / SAMPLE_RATE)
        q.put(None)

    threading.Thread(target=feed, daemon=True).start()
    while not stop.is_set() and (frame := q.get()) is not None:
        yield frame


class Display:
    def __init__(self):
        self.partial_shown = False

    def partial(self, text):
        width = shutil.get_terminal_size().columns - 4
        line = " ".join(text.split())
        line = ("…" + line[-(width - 1):]) if len(line) > width else line
        sys.stdout.write("\r\033[K\033[2m" + line + "\033[0m")
        sys.stdout.flush()
        self.partial_shown = True

    def final(self, text):
        if self.partial_shown:
            sys.stdout.write("\r\033[K")
        print(text, flush=True)
        self.partial_shown = False


LIVE_MODELS = ["medasr", "parakeet", "omni", "omni-urdu", "omni-1b", "omni-1b-urdu", "omni-ft", "dolphin-base", "dolphin-small",
               "whisper-small-urdu", "whisper-medium-urdu", "whisper-turbo-urdu",
               "whisper"]


def main():
    args = sys.argv[1:]
    source_file = args[args.index("--file") + 1] if "--file" in args else None

    if "--list-devices" in args:
        print_devices(list_input_devices(SAMPLE_RATE))
        return
    model = args[args.index("--model") + 1] if "--model" in args else "medasr"
    if model not in LIVE_MODELS:
        sys.exit(f"unknown model '{model}'. Choose one of: {', '.join(LIVE_MODELS)}")
    if not source_file:
        requested = args[args.index("--device") + 1] if "--device" in args else None
        device, device_name = choose_device(SAMPLE_RATE, requested)

    if model.startswith("omni"):
        print("loading Omnilingual ASR CTC (int8, CPU; Urdu + English) + VAD...", flush=True)
        asr = load_asr(model)
    elif model.startswith("dolphin"):
        print(f"loading {model} (int8 ONNX, CPU) + VAD...", flush=True)
        asr = load_asr(model)
    elif model.startswith("whisper-") and "-urdu" in model:
        runtime = "PyTorch fp32 -- slow on CPU" if model.endswith("-ft") else "int8 ONNX, CPU"
        print(f"loading {model} ({runtime}) + VAD...", flush=True)
        asr = load_asr(model)
    elif model == "parakeet":
        print("loading Parakeet TDT 0.6B (int8, CPU) + VAD...", flush=True)
        asr = load_asr("parakeet")
    elif model == "whisper":
        print(f"loading Whisper large-v3-turbo + VAD (cardiology prompt: "
              f"{'off' if '--no-lm' in args else 'on'})...", flush=True)
        asr = load_asr("whisper")
        if "--no-lm" in args:
            asr.prompt = ""
    else:
        lm_path = DEFAULT_LM_PATH if DEFAULT_LM_PATH.exists() and "--no-lm" not in args else None
        print(f"loading MedASR + VAD (cardiology LM: {'on' if lm_path else 'off'})...", flush=True)
        asr = load_asr("medasr", lm_path=lm_path)
    # Frequent re-transcription for live previews is only cheap enough with MedASR. Whisper always
    # encodes a fixed 30 s window (seconds per call on CPU); Parakeet is untested for this yet.
    # The others show a status line instead.
    live_preview = model == "medasr"
    vad = load_silero_vad()

    # Each session gets a folder: transcript.txt plus one wav/txt pair per finalized line,
    # which later become (corrected) training/evaluation data.
    session_dir = TRANSCRIPTS_DIR / f"live_{datetime.now():%Y%m%d_%H%M%S}"
    session_dir.mkdir(parents=True)
    out_path = session_dir / "transcript.txt"
    out = out_path.open("w", encoding="utf-8")

    stop = threading.Event()
    frames = file_frames(source_file, stop) if source_file else mic_frames(device, stop)
    display = Display()

    pre_roll = deque(maxlen=int(PRE_ROLL_MS / FRAME_MS))
    utterance, probs, speaking, silence_ms, since_partial = [], [], False, 0.0, 0.0
    latencies = []

    def finalize(frames):
        audio = np.concatenate(frames)
        if not live_preview:
            display.partial(f"[transcribing {len(audio) / SAMPLE_RATE:.1f}s of speech...]")
        t0 = time.perf_counter()
        text = asr.transcribe(audio)
        latencies.append(time.perf_counter() - t0)
        if text:
            display.final(text)
            out.write(text + "\n")
            out.flush()
            stem = session_dir / f"utt_{len(latencies):03d}"
            sf.write(stem.with_suffix(".wav"), audio, SAMPLE_RATE)
            stem.with_suffix(".txt").write_text(text, encoding="utf-8")

    source = f"streaming {source_file}" if source_file else f"listening on {device_name}"
    print(f"{source}... (Ctrl+C to stop)\n", flush=True)
    try:
        for frame in frames:
            prob = vad(torch.from_numpy(frame), SAMPLE_RATE).item()
            pre_roll.append(frame)  # always rolling, so a quick restart after a pause keeps its onset

            if not speaking:
                if prob >= SPEECH_PROB:
                    speaking, utterance = True, list(pre_roll)
                    probs = [1.0] * len(utterance)
                    silence_ms = since_partial = 0.0
                    if not live_preview:
                        display.partial("[recording... pause to transcribe]")
                continue

            utterance.append(frame)
            probs.append(prob)
            since_partial += FRAME / SAMPLE_RATE
            silence_ms = silence_ms + FRAME_MS if prob < SILENCE_PROB else 0.0
            utterance_s = len(utterance) * FRAME / SAMPLE_RATE

            if silence_ms >= END_SILENCE_MS:
                finalize(utterance)
                speaking, utterance, probs = False, [], []
            elif utterance_s >= MAX_UTTERANCE_S:
                # Still talking: cut at the quietest recent stretch (smoothed VAD), carry the rest over.
                search = int(SPLIT_SEARCH_S * SAMPLE_RATE / FRAME)
                smooth = np.convolve(probs[-search:], np.ones(7) / 7, mode="same")
                cut = len(probs) - search + int(np.argmin(smooth[3:-3])) + 3
                finalize(utterance[:cut])
                utterance, probs = utterance[cut:], probs[cut:]
                since_partial = 0.0
            elif live_preview and since_partial >= PARTIAL_EVERY_S and utterance_s >= 1.0:
                since_partial = 0.0
                if text := asr.transcribe(np.concatenate(utterance), use_lm=False):
                    display.partial(text)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        if utterance:
            finalize(utterance)
        out.close()
        if latencies:
            print(f"\nfinal-line latency after pause: avg {np.mean(latencies):.2f}s, "
                  f"max {np.max(latencies):.2f}s")
        print(f"transcript saved to {out_path}")


if __name__ == "__main__":
    main()
