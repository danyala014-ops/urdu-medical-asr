#!/usr/bin/env python3
"""Kaggle: fine-tune Meta Omnilingual ASR CTC 300M so Urdu speech comes out in Urdu script.

Why: the stock model writes spoken Urdu mostly in Hindi (Devanagari), which devanagari_to_urdu.py then
re-spells by rules. Fine-tuned on UrduSpeech labels (Urdu in Urdu script, English words in English
letters), the model should write Urdu as Urdu and English as English by itself.

The result stays laptop/CPU sized: same 300M model, exported to int8 ONNX with the same interface
as the sherpa-onnx model asr_engine.py already runs (input "x", logits out, same tokens.txt).

Steps:
1. Picks a random subset of UrduSpeech clips (code-switched, standard Urdu, Pakistani English;
   health-related clips first) and downloads only those clips -- never the full ~100 GB corpus
2. Voices synthetic doctor-patient sentences (medical_sentences.py, 264 English medical terms)
   with Meta's MMS Urdu and English TTS voices, so the model hears medical terms (diabetes,
   palpitations, ...) labelled in English letters
3. Adds real English medical speech from many speakers ("Medical Speech, Transcription, and
   Intent" Kaggle dataset), if it is attached as an input
4. Loads omniASR_CTC_300M -- or continues from a previous fine-tune (omni-urdu-ft-hf) if one is
   attached as an input -- and fine-tunes it (frozen conv front-end, speed/noise augmentation)
5. Reports before/after on hand-corrected real speech, unseen synthetic medical sentences,
   medical terms never seen in training, and real English medical speech
6. Exports model.int8.onnx + tokens.txt to /kaggle/working/omni-urdu-ft, and the full model
   (omni-urdu-ft-hf, ~1.3 GB) so the next round can continue training from it

How to run on Kaggle (no commands or arguments needed):
    1. Settings (right panel): Accelerator = GPU T4, Internet = On
    2. Add Input -> search "Medical Speech, Transcription, and Intent" (paultimothymooney) -> Add
       Optional, next rounds: Add Input -> Your Work -> this notebook's output (omni-urdu-ft-hf)
       Optional: Add-ons -> Secrets -> HF_TOKEN (avoids Hugging Face download rate limits)
    3. Paste this whole file into one notebook cell, then Save Version -> Save & Run All
    4. When it finishes, click the omni-urdu-ft.zip link under the output (or ask Claude to fetch
       the saved version's output with the Kaggle API if the browser download keeps failing)
To change how much data/training is used, edit the SETTINGS block below.
"""

import inspect
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
# The xet backend stalls ("reconstructing file 0%") on thousands of tiny wavs; plain HTTP is
# reliable. Also silence the per-file progress bars and fail fast instead of hanging.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
# Kaggle "T4 x2": avoid DataParallel; one T4 is plenty for a 300M model.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

for pkg, mod in [("onnx", "onnx"), ("onnxruntime", "onnxruntime")]:
    try:
        __import__(mod)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import HfApi, hf_hub_download
from transformers import Trainer, TrainingArguments, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC


DATASET_REPO = "ASLP-lab/UrduSpeech"
# HF Wav2Vec2ForCTC port of facebook/omniASR-CTC-300M; its vocab.json is token-for-token
# identical to the sherpa-onnx tokens.txt that asr_engine.py uses.
MODEL_ID = "ilhanemirhan/omniASR_CTC_300M"
REFERENCE_ONNX = "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12"

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
CKPT_DIR = WORK_DIR / "omni_ctc_checkpoints"
EXPORT_DIR = WORK_DIR / "omni-urdu-ft"
# Audio goes to /tmp on Kaggle so the gigabytes of clips never end up in the notebook's output.
DATA_DIR = (Path("/tmp") if WORK_DIR.as_posix() == "/kaggle/working" else WORK_DIR) / "urdu_speech_subset"
SYNTH_DIR = DATA_DIR.parent / "synthetic_medical"
MED_SPEECH_DIR = DATA_DIR.parent / "english_medical_speech"
FULL_MODEL_DIR = WORK_DIR / "omni-urdu-ft-hf"  # full-precision model, input for the next round
INPUT_DIR = Path("/kaggle/input")
# Meta MMS voices (CC-BY-NC-4.0): Urdu reads Urdu-script sentences, English reads English ones.
TTS_MODELS = {"urd": "facebook/mms-tts-urd-script_arabic", "eng": "facebook/mms-tts-eng"}
SENTENCES_URL = "https://raw.githubusercontent.com/danyala014-ops/urdu-medical-asr/main/medical_sentences.py"

# Share of training clips per subset. Code-switched speech teaches English words in English letters;
# standard Urdu got the larger share after run 1 (500 clips, 50% code-switched) started writing
# some plain Urdu words in English letters; Pakistani English keeps English in English letters.
TRAIN_MIX = {"US-Std": 0.5, "US-CS": 0.4, "US-EngPk": 0.1}
VAL_MIX = {"US-benchmark-CS": 0.5, "US-benchmark-Std": 0.35, "US-benchmark-EngPk": 0.15}
CLIP_FORMATS = ("short", "long")  # short <= 10 s, long 10-35 s (only clips <= MAX_SECONDS are used)
# Real clips mentioning health topics are taken first (English or Urdu words).
HEALTH_WORDS = re.compile(
    r"\b(doctor|hospital|medicines?|patients?|fever|cough|pain|blood|pressure|sugar|diabetes|heart|"
    r"infection|injection|tablets?|tests?|reports?|surgery|operation|treatment|disease|cancer|covid|"
    r"vaccine|health|healthy|symptoms|allergy|stress|anxiety|depression|clinic|nurse|emergency|bp|"
    r"ecg|ultrasound|vitamin|diet|exercise|brain|liver|kidney|lungs|stomach)\b"
    r"|ڈاکٹر|ہسپتال|دوائی|مریض|بخار|کھانسی|درد|بلڈ|شوگر|علاج|بیماری|انجکشن|آپریشن|صحت|خون|سانس|معدے|گردے|جگر",
    re.IGNORECASE)

# ============================== SETTINGS (edit here if needed) ==============================
TRAIN_CLIPS = 8000       # real UrduSpeech clips for training (~12 h of audio, downloaded to /tmp)
SYNTH_URDU = 5000        # synthetic Urdu doctor-patient sentences (Urdu voice) for training
SYNTH_ENGLISH = 1500     # synthetic English medical sentences (English voice) for training
SYNTH_TEST = 150         # unseen synthetic medical sentences (trained terms), only for measuring
UNSEEN_TEST = 100        # sentences built around 28 held-out terms never used in training
MED_SPEECH_TRAIN = 2000  # real English medical speech clips (needs the Kaggle input dataset)
MED_SPEECH_TEST = 150    # ... and held-out clips from its validation/test folders
VAL_CLIPS = 100          # hand-corrected benchmark clips used to measure the result
EPOCHS = 4
AUGMENT = True           # random speed (0.9-1.1x) and background noise on training audio
BATCH_SIZE = 2           # 4 ran out of T4 memory with 20 s clips; halved again automatically if needed
GRAD_ACCUM = 8           # BATCH_SIZE x GRAD_ACCUM = clips per training step (kept at 16)
LEARNING_RATE = 5e-5
CHECK_AGAINST_OFFICIAL_MODEL = True   # compare with Meta's official model before training
# ============================================================================================

MIN_CONFIDENCE = 0.95
MIN_SECONDS, MAX_SECONDS = 1.0, 20.0
SAMPLE_RATE = 16000
SEED = 42

BLANK_ID = 0  # "<s>": CTC blank, as in sherpa-onnx / asr_engine.py
SPECIAL_IDS = {0, 1, 2, 3}  # <s> <pad> </s> <unk>
# Urdu short-vowel marks, shadda, superscript alef, hamza above, tatweel: not in the vocab, and
# the stock model writes Urdu without them anyway.
DROP_CHARS = set("ًٌٍَُِّْٰٔـ")


def clean_text(text):
    if text is None:
        return ""
    return re.sub(r"\s+", " ", str(text)).strip()


def to_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


class CharVocab:
    """Character vocab of the model (id order == sherpa tokens.txt); token " " is the word gap."""

    def __init__(self, vocab_path):
        self.tok2id = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
        self.id2tok = {i: t for t, i in self.tok2id.items()}

    def normalize(self, text):
        """Lowercase English, drop diacritics, punctuation/unknown chars -> space."""
        out = []
        for ch in text.lower():
            if ch in DROP_CHARS:
                continue
            out.append(ch if ch in self.tok2id and ch.strip() else " ")
        return " ".join("".join(out).split())

    def encode(self, text):
        return [self.tok2id[ch] for ch in text]

    def ctc_decode(self, ids):
        out, prev = [], None
        for i in ids:
            i = int(i)
            if i != prev and i >= 0 and i not in SPECIAL_IDS:
                out.append(self.id2tok.get(i, ""))
            prev = i
        return " ".join("".join(out).split())

    def decode_labels(self, ids):
        return "".join(self.id2tok.get(int(i), "") for i in ids if i >= 0 and int(i) not in SPECIAL_IDS)

    def write_tokens_txt(self, path):
        lines = [f"{self.id2tok[i]} {i}" for i in range(len(self.id2tok))]
        Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def collect_candidates(repo_files, root, subset, text_keys, min_confidence, vocab):
    """Pair transcription JSONL rows with audio files that actually exist in the repo."""
    # Audio extensions are mixed case (.wav / .WAV), so match case-insensitively.
    audio_lookup = {f.lower(): f for f in repo_files if f.lower().endswith((".wav", ".flac"))}
    prefixes = [f"{root}/{subset}/{fmt}/" for fmt in CLIP_FORMATS]
    jsonl_files = [f for f in repo_files if f.startswith(tuple(prefixes)) and f.endswith("transcription.jsonl")]
    candidates = []
    for jsonl_rel in jsonl_files:
        jsonl_path = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=jsonl_rel)
        category_dir = jsonl_rel.rsplit("/", 1)[0]
        for row in read_jsonl(jsonl_path):
            text = ""
            for key in text_keys:
                text = vocab.normalize(clean_text(row.get(key)))
                if text:
                    break
            clip = row.get("Audio_Clip")
            if not text or not clip:
                continue
            duration = to_float(row.get("Duration_seconds"))
            if duration is not None and not (MIN_SECONDS <= duration <= MAX_SECONDS):
                continue
            confidence = to_float(row.get("Confidence_score"))
            if min_confidence and confidence is not None and confidence < min_confidence:
                continue
            audio_rel = audio_lookup.get(f"{category_dir}/audio/{clip}".lower())
            if audio_rel is None:
                continue
            raw = clean_text(row.get(text_keys[0]))
            candidates.append({"audio_rel": audio_rel, "text": text, "seconds": duration or 5.0,
                               "subset": subset, "health": bool(HEALTH_WORDS.search(raw))})
    n_health = sum(c["health"] for c in candidates)
    print(f"  {root}/{subset}: {len(candidates)} usable clips ({n_health} health-related)")
    return candidates


def sample_mix(repo_files, root, mix, total, text_keys, min_confidence, vocab, rng):
    selected = []
    for subset, share in mix.items():
        cands = collect_candidates(repo_files, root, subset, text_keys, min_confidence, vocab)
        rng.shuffle(cands)
        if root == "corpus":  # health-related clips first, the rest fills the quota
            cands.sort(key=lambda c: not c["health"])
        selected += cands[: max(1, round(total * share))]
    rng.shuffle(selected)
    return selected[:total]


def download_audio(rows, max_workers=16):
    """Download only the selected clips; drop rows whose download fails."""
    def fetch(row):
        for attempt in range(3):
            try:
                local = hf_hub_download(
                    repo_id=DATASET_REPO, repo_type="dataset", filename=row["audio_rel"], local_dir=str(DATA_DIR)
                )
                # Store as 16 kHz mono right away: ~4x smaller on disk and no resampling every epoch.
                small = Path(local).with_suffix(".16k.wav")
                if not small.exists():
                    sf.write(small, load_audio(local), SAMPLE_RATE, subtype="PCM_16")
                    Path(local).unlink(missing_ok=True)
                return {**row, "audio_path": str(small)}
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))

    out = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fetch, r) for r in rows]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                out.append(fut.result())
            except Exception as exc:
                print(f"  download failed: {exc}", file=sys.stderr)
            if i % 100 == 0 or i == len(rows):
                print(f"  downloaded {i}/{len(rows)}")
    return out


def build_splits(max_train, max_val, vocab):
    print(f"Listing files in {DATASET_REPO} ...")
    repo_files = HfApi().list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    rng = random.Random(SEED)

    print("Training candidates (corpus/):")
    train_sel = sample_mix(repo_files, "corpus", TRAIN_MIX, max_train, ["Transcription"],
                           MIN_CONFIDENCE, vocab, rng)
    print("Validation candidates (benchmark/, manually corrected):")
    val_sel = sample_mix(repo_files, "benchmark", VAL_MIX, max_val, ["ground_truth", "Transcription"],
                         None, vocab, rng)
    if not train_sel or not val_sel:
        raise RuntimeError("No usable clips found; the dataset layout may have changed.")

    total_min = sum(r["seconds"] for r in train_sel + val_sel) / 60
    print(f"Selected {len(train_sel)} train / {len(val_sel)} val clips "
          f"(~{total_min:.0f} min of audio); downloading only these ...")
    return download_audio(train_sel), download_audio(val_sel)


def load_sentence_module():
    """medical_sentences.py from the working folder, else from the project's GitHub repository."""
    try:
        import medical_sentences
        return medical_sentences
    except ImportError:
        import types
        import urllib.request

        code = urllib.request.urlopen(SENTENCES_URL, timeout=60).read().decode("utf-8")
        module = types.ModuleType("medical_sentences")
        exec(compile(code, SENTENCES_URL, "exec"), module.__dict__)
        return module


@torch.no_grad()
def synthesize_medical(vocab):
    """Voice synthetic doctor-patient sentences with the MMS Urdu and English voices.

    The Urdu voice reads the Urdu-script version of each sentence; the label keeps English terms
    in English letters. Speed, voice style and pitch vary per sentence. Returns (train_rows,
    seen_test_rows, unseen_test_rows); test sentences never appear in training, and the unseen
    test is built around held-out terms that no training sentence contains.
    """
    from transformers import AutoTokenizer, VitsModel

    sentences = load_sentence_module()
    train = sentences.generate(SYNTH_URDU, seed=1) + sentences.generate(SYNTH_ENGLISH, seed=11, voice="eng")
    labels = {x["label"] for x in train}
    seen = sentences.generate(SYNTH_TEST, seed=2, exclude=labels)
    unseen = (sentences.generate(UNSEEN_TEST * 7 // 10, seed=3, held_out="only", exclude=labels)
              + sentences.generate(UNSEEN_TEST - UNSEEN_TEST * 7 // 10, seed=4, held_out="only",
                                   voice="eng", exclude=labels))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    voices = {v: (AutoTokenizer.from_pretrained(m), VitsModel.from_pretrained(m).to(device).eval())
              for v, m in TTS_MODELS.items()}
    SYNTH_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    torch.manual_seed(SEED)
    print(f"Voicing {len(train)} training + {len(seen)} + {len(unseen)} test sentences "
          f"({len(sentences.all_terms())} English medical terms) ...")

    def voice(items, name):
        rows = []
        for i, item in enumerate(items):
            tokenizer, tts = voices[item["voice"]]
            text = "".join(ch for ch in item["tts"] if ch not in DROP_CHARS)
            tts.speaking_rate = rng.uniform(0.85, 1.15)
            tts.noise_scale = rng.uniform(0.45, 0.8)
            wav = tts(**tokenizer(text, return_tensors="pt").to(device)).waveform[0].float().cpu().numpy()
            if tts.config.sampling_rate != SAMPLE_RATE:
                wav = librosa.resample(wav, orig_sr=tts.config.sampling_rate, target_sr=SAMPLE_RATE)
            if rng.random() < 0.7:  # one speaker per voice, so vary the pitch to sound like more people
                wav = librosa.effects.pitch_shift(wav, sr=SAMPLE_RATE, n_steps=rng.uniform(-3, 3))
            path = SYNTH_DIR / f"{name}_{i:05d}.wav"
            sf.write(path, wav.astype(np.float32), SAMPLE_RATE)
            rows.append({"audio_path": str(path), "text": vocab.normalize(item["label"]), "terms": item["terms"],
                         "seconds": len(wav) / SAMPLE_RATE, "subset": f"synthetic-{item['voice']}"})
            if (i + 1) % 1000 == 0:
                print(f"  {name}: {i + 1}/{len(items)}")
        return rows

    out = voice(train, "train"), voice(seen, "seen"), voice(unseen, "unseen")
    del voices
    torch.cuda.empty_cache()
    return out


def load_medical_speech(vocab):
    """Real English medical speech (many speakers) from the attached Kaggle dataset, if any.

    "Medical Speech, Transcription, and Intent" (Figure Eight): ~8.5 h of people describing
    symptoms. Clips are filtered by the dataset's own audio-quality rating and stored as 16 kHz.
    Returns (train_rows, test_rows); test rows come from its validate/test folders.
    """
    import csv

    csvs = list(INPUT_DIR.rglob("overview-of-recordings.csv")) if INPUT_DIR.exists() else []
    if not csvs:
        print("English medical speech dataset not attached (Add Input -> 'Medical Speech, "
              "Transcription, and Intent'); continuing without it.")
        return [], []
    root = csvs[0].parent
    wav_paths = {p.name: p for p in root.rglob("*.wav")}
    train, test = [], []
    with open(csvs[0], encoding="utf-8", errors="ignore") as f:
        for row in csv.DictReader(f):
            path = wav_paths.get((row.get("file_name") or "").strip())
            text = vocab.normalize(clean_text(row.get("phrase")))
            quality = to_float(row.get("overall_quality_of_the_audio"), 0.0)
            if path is None or not text or quality < 3.5:
                continue
            (train if path.parent.name == "train" else test).append({"src": path, "text": text, "terms": []})
    rng = random.Random(SEED)
    rng.shuffle(train)
    rng.shuffle(test)
    train, test = train[:MED_SPEECH_TRAIN], test[:MED_SPEECH_TEST]
    MED_SPEECH_DIR.mkdir(parents=True, exist_ok=True)

    def convert(row):
        dst = MED_SPEECH_DIR / (row["src"].stem + ".16k.wav")
        wav = load_audio(row["src"])
        sf.write(dst, wav, SAMPLE_RATE, subtype="PCM_16")
        return {"audio_path": str(dst), "text": row["text"], "terms": [], "seconds": len(wav) / SAMPLE_RATE,
                "subset": "english-medical"}

    with ThreadPoolExecutor(max_workers=8) as pool:
        train, test = list(pool.map(convert, train)), list(pool.map(convert, test))
    print(f"English medical speech: {len(train)} training + {len(test)} test clips")
    return train, test


def find_previous_model():
    """A full fine-tuned model attached as input (omni-urdu-ft-hf from an earlier round), if any."""
    if not INPUT_DIR.exists():
        return None
    for config in INPUT_DIR.rglob("config.json"):
        folder = config.parent
        if (folder / "model.safetensors").exists() and '"wav2vec2"' in config.read_text(encoding="utf-8"):
            return folder
    return None


def load_audio(path):
    waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return waveform.astype(np.float32)


def filter_readable(rows):
    """Drop unreadable, too-short or non-finite clips once up front so training never crashes."""
    good = []
    for row in rows:
        try:
            wav = load_audio(row["audio_path"])
        except Exception:
            continue
        if wav.size < SAMPLE_RATE // 2 or not np.all(np.isfinite(wav)):
            continue
        good.append(row)
    return good


class SpeechDataset(torch.utils.data.Dataset):
    def __init__(self, rows, augment=False):
        self.rows = rows
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return {**self.rows[idx], "augment": self.augment}


def augment_audio(wav, rng=np.random):
    """Random speed change (0.9-1.1x) and background noise (SNR 10-30 dB), each half the time."""
    if rng.random() < 0.5:
        factor = rng.uniform(0.9, 1.1)
        wav = librosa.resample(wav, orig_sr=SAMPLE_RATE, target_sr=int(SAMPLE_RATE / factor))
    if rng.random() < 0.5:
        power = float(np.mean(wav ** 2)) + 1e-10
        snr_db = rng.uniform(10, 30)
        wav = wav + rng.normal(0, np.sqrt(power / 10 ** (snr_db / 10)), wav.shape).astype(np.float32)
    return wav.astype(np.float32)


class CTCCollator:
    """Loads audio, normalises/pads it and pads character labels per batch (lazy, low RAM)."""

    def __init__(self, feature_extractor, vocab):
        self.feature_extractor = feature_extractor
        self.vocab = vocab

    def __call__(self, rows):
        audios = [load_audio(r["audio_path"]) for r in rows]
        audios = [augment_audio(a) if AUGMENT and r.get("augment") else a for a, r in zip(audios, rows)]
        batch = self.feature_extractor(
            audios, sampling_rate=SAMPLE_RATE,
            padding=True, return_attention_mask=True, return_tensors="pt",
        )
        labels = [self.vocab.encode(r["text"]) for r in rows]
        width = max(len(l) for l in labels)
        padded = torch.full((len(labels), width), -100, dtype=torch.long)
        for i, l in enumerate(labels):
            padded[i, : len(l)] = torch.tensor(l, dtype=torch.long)
        return {"input_values": batch["input_values"], "attention_mask": batch["attention_mask"],
                "labels": padded}


def edit_distance(ref, hyp):
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1]


def error_rates(refs, hyps):
    w_err = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    w_tot = sum(len(r.split()) for r in refs) or 1
    c_err = sum(edit_distance(r, h) for r, h in zip(refs, hyps))
    c_tot = sum(len(r) for r in refs) or 1
    return w_err / w_tot, c_err / c_tot


def script_shares(texts):
    """Share of letters written in Urdu/Arabic script, Devanagari (Hindi) and Latin (English)."""
    counts = {"urdu": 0, "hindi": 0, "english": 0}
    for t in texts:
        for ch in t:
            o = ord(ch)
            if 0x0600 <= o <= 0x06FF:
                counts["urdu"] += 1
            elif 0x0900 <= o <= 0x097F:
                counts["hindi"] += 1
            elif "a" <= ch <= "z":
                counts["english"] += 1
    total = sum(counts.values()) or 1
    return {k: v / total for k, v in counts.items()}


def make_compute_metrics(vocab):
    def compute_metrics(pred):
        hyps = [vocab.ctc_decode(p) for p in pred.predictions]
        refs = [vocab.decode_labels(l) for l in pred.label_ids]
        wer, cer = error_rates(refs, hyps)
        shares = script_shares(hyps)
        return {"wer": wer, "cer": cer, "hindi_share": shares["hindi"], "urdu_share": shares["urdu"],
                "english_share": shares["english"]}
    return compute_metrics


def argmax_logits(logits, labels):
    # Keep eval memory small: frames x 9812 logits -> frame ids.
    return logits.argmax(-1)


def onnx_session(path):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, (os.cpu_count() or 2))
    return ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])


def onnx_transcribe(session, vocab, rows):
    """Same preprocessing + greedy CTC as asr_engine.OmnilingualASR (without the Urdu rules)."""
    hyps = []
    for r in rows:
        audio = load_audio(r["audio_path"])
        x = (audio - audio.mean()) / (audio.std() + 1e-7)
        logits = session.run(None, {"x": x[None].astype(np.float32)})[0][0]
        hyps.append(vocab.ctc_decode(logits.argmax(-1)))
    return hyps


@torch.no_grad()
def torch_transcribe(model, feature_extractor, vocab, rows):
    device = next(model.parameters()).device
    hyps = []
    for r in rows:
        inputs = feature_extractor(load_audio(r["audio_path"]), sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = model(inputs.input_values.to(device)).logits[0]
        hyps.append(vocab.ctc_decode(logits.argmax(-1).cpu().numpy()))
    return hyps


def medical_report(model, feature_extractor, vocab, rows):
    """WER/CER on unseen synthetic medical sentences + share of English terms written exactly."""
    hyps = torch_transcribe(model, feature_extractor, vocab, rows)
    wer, cer = error_rates([r["text"] for r in rows], hyps)
    found = total = 0
    for r, h in zip(rows, hyps):
        for term in r["terms"]:
            total += 1
            found += bool(re.search(rf"(?<![a-z]){re.escape(vocab.normalize(term))}(?![a-z])", h))
    return {"wer": wer, "cer": cer, "terms": found / total if total else None,
            "example": (rows[0]["text"], hyps[0])}


def check_against_reference(model, feature_extractor, vocab, rows):
    """The HF port must transcribe like the official model before we train it."""
    try:
        ref_path = hf_hub_download(REFERENCE_ONNX, "model.int8.onnx")
    except Exception as exc:
        print(f"(skipping reference check, could not download reference model: {exc})")
        return
    ref_hyps = onnx_transcribe(onnx_session(ref_path), vocab, rows)
    hf_hyps = torch_transcribe(model, feature_extractor, vocab, rows)
    _, diff = error_rates(ref_hyps, hf_hyps)
    print(f"\nHF port vs official int8 ONNX on {len(rows)} clips: {diff:.1%} character difference")
    print(f"  official: {ref_hyps[0]}\n  HF port : {hf_hyps[0]}")
    if diff > 0.25:
        raise RuntimeError("The HF port does not match the official model; not training on it.")


class ExportWrapper(torch.nn.Module):
    """ONNX signature of the sherpa-onnx model: x (N, T) normalised audio -> logits (N, T', V)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(input_values=x).logits


def export_onnx(model, vocab, val_rows):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    model = model.float().cpu().eval()
    # Bake weight-norm into a plain conv weight so the ONNX graph is simple.
    pos_conv = model.wav2vec2.encoder.pos_conv_embed.conv
    if hasattr(pos_conv, "parametrizations"):
        torch.nn.utils.parametrize.remove_parametrizations(pos_conv, "weight")

    fp32_path = WORK_DIR / "omni_ft_fp32.onnx"
    int8_path = EXPORT_DIR / "model.int8.onnx"
    dummy = torch.randn(1, SAMPLE_RATE * 4)
    export_kwargs = dict(
        input_names=["x"], output_names=["logits"],
        dynamic_axes={"x": {0: "N", 1: "T"}, "logits": {0: "N", 1: "T_out"}},
        opset_version=17, do_constant_folding=True,
    )
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_kwargs["dynamo"] = False  # classic exporter: dynamic audio length works reliably
    print("\nExporting ONNX ...")
    torch.onnx.export(ExportWrapper(model), (dummy,), str(fp32_path), **export_kwargs)
    # Only MatMuls (the transformer) go int8, like a typical int8 CTC export; convs stay fp32.
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8,
                     op_types_to_quantize=["MatMul"])
    fp32_path.unlink(missing_ok=True)
    vocab.write_tokens_txt(EXPORT_DIR / "tokens.txt")
    size_mb = int8_path.stat().st_size / 1e6
    print(f"Saved {int8_path} ({size_mb:.0f} MB) + tokens.txt")

    # Check the exported file exactly the way the laptop app will run it.
    rows = val_rows[:30]
    hyps = onnx_transcribe(onnx_session(int8_path), vocab, rows)
    wer, cer = error_rates([r["text"] for r in rows], hyps)
    shares = script_shares(hyps)
    print(f"int8 ONNX on {len(rows)} benchmark clips: WER {wer:.3f}  CER {cer:.3f}  "
          f"(Urdu {shares['urdu']:.0%} / Hindi {shares['hindi']:.0%} / English {shares['english']:.0%} of letters)")
    for r, h in list(zip(rows, hyps))[:4]:
        print(f"  REF: {r['text']}\n  OUT: {h}\n")


def load_hf_token():
    """Use an HF_TOKEN saved in Kaggle Secrets if there is one (faster downloads); optional."""
    if os.environ.get("HF_TOKEN"):
        return
    try:
        from kaggle_secrets import UserSecretsClient

        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        print("Using HF_TOKEN from Kaggle Secrets")
    except Exception:
        print("No HF_TOKEN secret found; continuing without it (that's fine, the data is public)")


def make_trainer(model, train_rows, val_rows, med_rows, feature_extractor, vocab, batch_size, grad_accum):
    steps_per_epoch = math.ceil(len(train_rows) / (batch_size * grad_accum))
    total_steps = max(1, math.ceil(steps_per_epoch * EPOCHS))
    print(f"Training: batch {batch_size} x {grad_accum} accumulation, ~{total_steps} optimizer steps")

    arg_kwargs = dict(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=LEARNING_RATE,
        warmup_steps=max(1, total_steps // 10),
        num_train_epochs=EPOCHS,
        weight_decay=0.0,
        save_strategy="epoch",
        logging_steps=10,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="real_cer",  # the hand-corrected benchmark decides the best epoch
        greater_is_better=False,
        # No gradient checkpointing: with Wav2Vec2 on recent transformers/torch it fails in backward
        # ("different number of tensors saved during forward and recomputation"). A 300M model on
        # <=20 s clips fits a 16 GB T4 without it at batch 2.
        gradient_checkpointing=False,
        fp16=torch.cuda.is_available(),
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=2,
        seed=SEED,
    )
    # transformers renamed evaluation_strategy -> eval_strategy (old name removed in 4.46+).
    if "eval_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        arg_kwargs["eval_strategy"] = "epoch"
    else:
        arg_kwargs["evaluation_strategy"] = "epoch"

    return Trainer(
        model=model,
        args=TrainingArguments(**arg_kwargs),
        train_dataset=SpeechDataset(train_rows, augment=True),
        eval_dataset={"real": SpeechDataset(val_rows), "medical": SpeechDataset(med_rows[:150])},
        data_collator=CTCCollator(feature_extractor, vocab),
        compute_metrics=make_compute_metrics(vocab),
        preprocess_logits_for_metrics=argmax_logits,
    )


def train_with_memory_fallback(model, train_rows, val_rows, med_rows, feature_extractor, vocab):
    """Train; if the GPU runs out of memory, halve the batch (same clips per step) and retry."""
    batch_size, grad_accum = BATCH_SIZE, GRAD_ACCUM
    before = None
    while True:
        trainer = make_trainer(model, train_rows, val_rows, med_rows, feature_extractor, vocab,
                               batch_size, grad_accum)
        try:
            if before is None:  # measure the untouched model once; a retry starts from trained weights
                before = trainer.evaluate(trainer.eval_dataset["real"], metric_key_prefix="eval_real")
            trainer.train()
            return trainer, before
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            del trainer
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            shutil.rmtree(CKPT_DIR, ignore_errors=True)
            batch_size, grad_accum = batch_size // 2, grad_accum * 2
            print(f"GPU out of memory -> retrying with batch {batch_size} x {grad_accum} accumulation")


def main():
    load_hf_token()
    for d in (CKPT_DIR, EXPORT_DIR, DATA_DIR, SYNTH_DIR, MED_SPEECH_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print(f"CUDA available: {torch.cuda.is_available()}"
          + (f" ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else ""))
    if not torch.cuda.is_available():
        print("WARNING: no GPU. In Kaggle: Settings -> Accelerator -> GPU T4, then run again.")

    local_vocab = Path(MODEL_ID) / "vocab.json"
    vocab_path = local_vocab if local_vocab.exists() else hf_hub_download(MODEL_ID, "vocab.json")
    vocab = CharVocab(vocab_path)
    train_rows, val_rows = build_splits(TRAIN_CLIPS, VAL_CLIPS, vocab)
    train_rows, val_rows = filter_readable(train_rows), filter_readable(val_rows)
    synth_train, seen_rows, unseen_rows = synthesize_medical(vocab)
    speech_train, speech_test = load_medical_speech(vocab)
    train_rows = train_rows + synth_train + speech_train
    random.Random(SEED).shuffle(train_rows)
    print(f"Training clips: {len(train_rows)} ({len(synth_train)} synthetic medical, "
          f"{len(speech_train)} real English medical); validation: {len(val_rows)} real")
    if not train_rows or not val_rows:
        raise RuntimeError("Training set or validation set is empty after filtering.")

    previous = find_previous_model()
    start_from = str(previous) if previous else MODEL_ID
    print(f"Starting from: {start_from}" + ("  (continuing an earlier fine-tune)" if previous else ""))
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(
        start_from,
        ctc_loss_reduction="mean",
        ctc_zero_infinity=True,      # skip impossible (too-long label) clips instead of NaN loss
        pad_token_id=BLANK_ID,       # HF uses pad_token_id as the CTC blank
        layerdrop=0.0,
        apply_spec_augment=True,
        mask_time_prob=0.05,
        mask_feature_prob=0.0,
    )
    model.freeze_feature_encoder()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if CHECK_AGAINST_OFFICIAL_MODEL and not previous:
        check_against_reference(model, feature_extractor, vocab, val_rows[:5])
    tests = {"Synthetic medical sentences (trained terms)": seen_rows,
             "Synthetic medical sentences (28 terms never trained on)": unseen_rows,
             "Real English medical speech (many speakers)": speech_test}
    tests = {k: v for k, v in tests.items() if v}
    med_before = {k: medical_report(model, feature_extractor, vocab, v) for k, v in tests.items()}

    trainer, before = train_with_memory_fallback(model, train_rows, val_rows, seen_rows, feature_extractor, vocab)
    after = trainer.evaluate(trainer.eval_dataset["real"], metric_key_prefix="eval_real")
    trainer.model.eval()
    med_after = {k: medical_report(trainer.model, feature_extractor, vocab, v) for k, v in tests.items()}

    start_name = "previous fine-tune" if previous else "stock omniASR 300M"
    print("\nBenchmark clips (manually corrected Urdu/English), greedy CTC:")
    for name, m in [(start_name, before), ("fine-tuned", after)]:
        print(f"  {name:<19} WER {m['eval_real_wer']:.3f}  CER {m['eval_real_cer']:.3f}  "
              f"letters: Urdu {m['eval_real_urdu_share']:.0%} / Hindi {m['eval_real_hindi_share']:.0%} "
              f"/ English {m['eval_real_english_share']:.0%}")
    for title, rows in tests.items():
        print(f"\n{title} ({len(rows)}):")
        for name, m in [(start_name, med_before[title]), ("fine-tuned", med_after[title])]:
            terms = f"  medical terms exactly right {m['terms']:.0%}" if m["terms"] is not None else ""
            print(f"  {name:<19} WER {m['wer']:.3f}  CER {m['cer']:.3f}{terms}")
        print(f"  e.g. REF: {med_after[title]['example'][0]}\n       OUT: {med_after[title]['example'][1]}")

    # Full-precision model first (export_onnx modifies the model): the next round continues from it.
    trainer.model.save_pretrained(str(FULL_MODEL_DIR))
    feature_extractor.save_pretrained(str(FULL_MODEL_DIR))
    shutil.copy(vocab_path, FULL_MODEL_DIR / "vocab.json")
    print(f"\nSaved full model for the next round: {FULL_MODEL_DIR}")
    export_onnx(trainer.model, vocab, val_rows)
    shutil.rmtree(CKPT_DIR, ignore_errors=True)  # checkpoints are ~3.6 GB each; the ONNX is what we keep
    shutil.rmtree(DATA_DIR, ignore_errors=True)  # audio clips are not needed any more
    shutil.rmtree(SYNTH_DIR, ignore_errors=True)
    shutil.rmtree(MED_SPEECH_DIR, ignore_errors=True)

    # Kaggle's Output panel often fails to start big downloads; a zip + direct link is reliable.
    zip_path = shutil.make_archive(str(EXPORT_DIR), "zip", str(EXPORT_DIR))
    print(f"\nDone. Click the link below to download {Path(zip_path).name}, then unzip it into the "
          "project's models/omni-urdu-ft/ folder (model.int8.onnx + tokens.txt).")
    try:
        from IPython.display import FileLink, display

        display(FileLink(os.path.relpath(zip_path, Path.cwd())))
    except ImportError:
        print(zip_path)


if __name__ == "__main__":
    main()
