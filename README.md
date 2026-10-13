# Urdu Medical ASR

Speech-to-text for doctor–patient conversations in **Urdu mixed with English**, running fully
offline on an ordinary laptop CPU.

Urdu speech is written in **Urdu script** and English words (medical terms, everyday English) in
**English letters**:

> ڈاکٹر صاحب، مجھے **chest pain** ہو رہا ہے جو **left arm** تک جاتا ہے

The model is Meta's [Omnilingual ASR CTC 300M](https://huggingface.co/facebook/omniASR-CTC-300M),
fine-tuned on public Urdu speech and exported to an int8 ONNX file (365 MB) that transcribes about
4× faster than real time on a laptop CPU (Intel i5-10310U, no GPU).

## Why fine-tune?

The stock Omnilingual model recognises Urdu well but writes it mostly in **Hindi (Devanagari)**
script, because spoken Urdu and Hindi sound almost the same, and it writes English words in Urdu
or Hindi script. The earlier workaround (`omni-urdu`) re-spells the output into Urdu with rules
(`devanagari_to_urdu.py`). The fine-tuned model (`omni-ft`) writes Urdu and English correctly by itself.

## Results

WER = word error rate, CER = character error rate (lower is better).

| Test set | Stock + Urdu rules (`omni-urdu`) | **Fine-tuned (`omni-ft`)** |
|---|---|---|
| Urdu + English, everyday speech (30 UrduSpeech clips) | WER 58.0% · CER 42.6% | **WER 39.2% · CER 23.1%** |
| 15 read doctor–patient sentences (own voice) | WER 29% · CER 23% | **WER 20% · CER 10%** |
| Formal pure Urdu (30 FLEURS clips) | WER 34.6% · CER **12.1%** | **WER 32.6%** · CER 15.9% |

On formal pure Urdu the fine-tune writes English loanwords such as *modern* or *infection* in
English letters while the references use Urdu script, which counts as character errors.

Known weaknesses: some English medical terms are spelled by sound (*dibeties*, *medisn*), and
aspirated Urdu letters are sometimes dropped (کھانسی → خانسی).

## Setup

Python 3.12+ on Linux or Windows.

```bash
git clone https://github.com/danyala014-ops/urdu-medical-asr.git
cd urdu-medical-asr
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Get the fine-tuned model** (too large for the repository): download `omni-urdu-ft.zip` from
this repository's **Releases** page and unzip it so that you have

```
models/omni-urdu-ft/model.int8.onnx
models/omni-urdu-ft/tokens.txt
```

Other models (`omni`, `omni-urdu`, Whisper, …) are downloaded from Hugging Face with
`python download_model.py <repo-id>`; the error message tells you the exact command.

## Usage

```bash
python live_transcribe.py --model omni-ft                 # live microphone, Ctrl+C to stop
python live_transcribe.py --list-devices                  # list microphones
python live_transcribe.py --model omni-ft --device 2      # use microphone #2
python live_transcribe.py --model omni-ft --file clip.wav # transcribe a recording
python benchmark.py tests/urduspeech_codeswitched --models omni-urdu,omni-ft   # WER/CER
```

Live sessions are saved under `transcripts/` (audio + text per sentence; not tracked by git).

## Training

`train_omni_kaggle.py` trains the model on a free Kaggle GPU:

1. Kaggle notebook settings: **Accelerator = GPU T4**, **Internet = On**.
2. **Add Input** → search *Medical Speech, Transcription, and Intent* (paultimothymooney) → Add.
   To continue from an earlier run, also add that notebook's output (it contains `omni-urdu-ft-hf/`).
   Optional: **Add-ons → Secrets → `HF_TOKEN`** avoids Hugging Face download rate limits.
3. Paste the whole file into one cell and click **Save Version → Save & Run All** (about 4 hours).
4. Download `omni-urdu-ft.zip` from the link printed at the end.

Training data:

- **Real Urdu/English speech:** 8,000 clips from [UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech)
  (50% standard Urdu, 40% Urdu–English code-switched, 10% Pakistani English), health-related
  clips first. Only these clips are downloaded, never the whole corpus.
- **Synthetic medical speech:** 6,500 doctor–patient sentences from `medical_sentences.py`
  (264 English medical terms), voiced with Meta's MMS Urdu and English text-to-speech. The Urdu
  voice reads English terms written in Urdu script (so it says *diabetes* the way Pakistani
  doctors do) while the label keeps them in English letters, so the model itself learns to
  write `diabetes`, `palpitations`, `angiography`, …
- **Real English medical speech:** 2,000 clips of many people describing symptoms
  (Kaggle *Medical Speech, Transcription, and Intent*, Figure Eight).
- **Augmentation:** random speed (0.9–1.1×) and background noise.

The script reports before/after results on hand-corrected real speech, unseen synthetic medical
sentences, 28 medical terms held out of training (to measure spelling of never-heard words) and
real English medical speech. It exports `model.int8.onnx` + `tokens.txt` (what `asr_engine.py`
loads) and the full-precision model `omni-urdu-ft-hf/`, from which the next round continues.
The notebook fetches `medical_sentences.py` from this repository, so push changes to it first.

## Project structure

| File | Purpose |
|---|---|
| `asr_engine.py` | Loads any model by name (`load_asr("omni-ft")`) and transcribes audio; post-processing for the fine-tune (one script per word, filler removal) |
| `live_transcribe.py` | Live microphone transcription with voice-activity detection |
| `benchmark.py` | Scores models on a test folder (WER, CER, Urdu CER, English words kept, speed) |
| `train_omni_kaggle.py` | Fine-tunes Omnilingual 300M on Kaggle and exports the int8 ONNX model |
| `medical_sentences.py` | Generates synthetic doctor–patient sentences (Urdu + English medical terms) for training |
| `devanagari_to_urdu.py` | Re-spells Hindi-script output into Urdu script (used by `omni-urdu`) |
| `english_terms.py` + `vocab/english_terms.txt` | Medical English terms written in Urdu script → English letters |
| `clinical_extraction.py` | Rules that pull symptoms, diagnoses, medicines and tests out of a transcript and store them in SQLite |
| `microphones.py` | Microphone listing and selection |
| `download_model.py` | Downloads a Hugging Face model with a progress readout |
| `tests/` | Two 30-clip test sets with reference transcripts (see `tests/README.md`) |

## Data and licences

- Code in this repository: [MIT](LICENSE).
- Base model: Meta Omnilingual ASR CTC 300M, Apache-2.0
  (weights via the Hugging Face port `ilhanemirhan/omniASR_CTC_300M`).
- Training and code-switched test data: UrduSpeech, CC-BY-4.0 (Haq et al., 2026, arXiv:2605.17846).
- Pure-Urdu test data: Google FLEURS, CC-BY-4.0.
- Synthetic training speech: Meta MMS Urdu and English TTS (`facebook/mms-tts-urd-script_arabic`,
  `facebook/mms-tts-eng`), CC-BY-NC-4.0, so models trained with it are for non-commercial use.
- Real English medical speech: Kaggle *Medical Speech, Transcription, and Intent* (Figure Eight);
  no explicit licence is given, so it is used for training only and not redistributed.
- No patient or hospital audio is used anywhere in this project.

The fine-tuned model is a derivative of the Apache-2.0 base model trained on CC-BY-4.0 data: when
you redistribute it, keep the Apache-2.0 notice and credit UrduSpeech.
