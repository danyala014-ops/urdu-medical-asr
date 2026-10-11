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

`train_omni_kaggle.py` re-creates the fine-tuned model on a free Kaggle GPU:

1. Kaggle notebook settings: **Accelerator = GPU T4**, **Internet = On**.
2. Paste the whole file into one cell and press **Run All** (about 2.5 hours).
3. Download `omni-urdu-ft.zip` from the link printed at the end.

It uses three kinds of training data:

- **Real speech:** 8,000 clips from [UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech)
  (50% standard Urdu, 40% Urdu–English code-switched, 10% Pakistani English), with the ~1,500
  health-related clips taken first. Only these clips are downloaded, never the whole corpus.
- **Synthetic medical speech:** 2,500 doctor–patient sentences from `medical_sentences.py`, voiced
  with Meta's MMS Urdu text-to-speech. The voice reads English terms written in Urdu script (so it
  says *diabetes* the way Pakistani doctors do) while the training label keeps them in English
  letters, so the model itself learns to write `diabetes`, `palpitations`, `angiography`, …
- **Augmentation:** random speed (0.9–1.1×) and background noise on the training audio.

It reports the error rates before and after training on hand-corrected real speech and on 150
unseen medical sentences (including the share of medical terms written exactly right), then
exports `model.int8.onnx` + `tokens.txt` in exactly the format `asr_engine.py` loads. Settings
are at the top of the file. The Kaggle notebook fetches `medical_sentences.py` from this
repository, so push changes to it before training.

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
- Synthetic training speech: Meta MMS Urdu TTS (`facebook/mms-tts-urd-script_arabic`), CC-BY-NC-4.0.
- No patient or hospital audio is used anywhere in this project.

The fine-tuned model is a derivative of the Apache-2.0 base model trained on CC-BY-4.0 data: when
you redistribute it, keep the Apache-2.0 notice and credit UrduSpeech.
