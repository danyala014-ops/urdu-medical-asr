"""Speech recognition engine shared by all scripts: loads a model by name and transcribes audio.

load_asr(name) returns an object with .transcribe(audio_16k) -> text. Main models:
  omni-ft    our fine-tuned Meta Omnilingual CTC 300M (models/omni-urdu-ft/, int8 ONNX, CPU):
             Urdu in Urdu script, English words in English letters
  omni-urdu  stock Omnilingual 300M + Hindi->Urdu re-spelling (devanagari_to_urdu.py)
Also: Google MedASR (English, optional cardiology LM), Whisper, Parakeet, Dolphin.
"""

import logging
import os
import re
import warnings
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
logging.getLogger("pyctcdecode").setLevel(logging.ERROR)  # "kenlm not installed": expected, we use ngram_lm
warnings.filterwarnings("ignore", message="Using padding='same'")  # harmless PyTorch conv notice

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor

MODEL_ID = "google/medasr"
SAMPLE_RATE = 16000
DEFAULT_LM_PATH = Path(__file__).parent / "lm" / "cath_lm.pkl.gz"
MAX_SEGMENT_S = 25  # longer audio is split at quiet points before decoding

# MedASR emits spoken-dictation formatting tokens instead of punctuation.
FORMAT_TOKENS = {
    "{period}": ".",
    "{comma}": ",",
    "{colon}": ":",
    "{semicolon}": ";",
    "{question mark}": "?",
    "{exclamation point}": "!",
    "{hyphen}": "-",
    "{new line}": "\n",
    "{new paragraph}": "\n\n",
}


def clean_transcript(raw):
    text = raw.replace("</s>", "")
    for token, repl in FORMAT_TOKENS.items():
        text = text.replace(" " + token, repl).replace(token, repl)
    text = re.sub(r"\s*(\[[A-Z ]+\])", r"\n\1", text)  # section headers on their own line
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n\s+", "\n", text).strip()


def piece_to_word(piece):
    """SentencePiece piece -> the "word" unit shared by the decoder and the n-gram LM."""
    return piece.replace("▁", "#")


def split_long_audio(audio, max_s=MAX_SEGMENT_S, search_s=5):
    """Split audio into <= max_s segments, cutting at the quietest 30 ms in the last search_s."""
    max_len, frame = int(max_s * SAMPLE_RATE), int(0.03 * SAMPLE_RATE)
    segments = []
    while len(audio) > max_len:
        window = audio[max_len - int(search_s * SAMPLE_RATE):max_len]
        energy = np.convolve(window ** 2, np.ones(frame), mode="valid")
        cut = max_len - len(window) + int(np.argmin(energy)) + frame // 2
        segments.append(audio[:cut])
        audio = audio[cut:]
    segments.append(audio)
    return segments


class MedASR:
    """MedASR with optional n-gram LM beam search (see build_lm.py)."""

    # alpha/beta tuned on a synthetic PCI clip + Google's radiology sample; re-tune on real
    # cath-lab recordings with `medasr_test.py --compare` once available.
    def __init__(self, lm_path=None, alpha=0.5, beta=1.0, beam_width=8):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForCTC.from_pretrained(MODEL_ID).to(self.device).eval()
        self.beam_width = beam_width
        self.decoder = self._build_decoder(lm_path, alpha, beta) if lm_path else None

    def _build_decoder(self, lm_path, alpha, beta):
        from pyctcdecode import Alphabet, BeamSearchDecoderCTC

        from ngram_lm import NgramLM, PyctcdecodeNgramLM

        tok = self.processor.tokenizer
        labels = [None] * tok.vocab_size
        for piece, idx in tok.get_vocab().items():
            if idx < tok.vocab_size:
                labels[idx] = piece
        # As in Google's MedASR notebook: blank -> "", and every normal piece becomes its own
        # "word" (prefix "▁", real word boundaries "▁" -> "#").
        labels[0] = ""
        labels = [p if i == 0 or (p.startswith("<") and p.endswith(">"))
                  else "▁" + piece_to_word(p) for i, p in enumerate(labels)]
        lm = PyctcdecodeNgramLM(NgramLM.load(lm_path), alpha=alpha, beta=beta)
        return BeamSearchDecoderCTC(Alphabet.build_alphabet(labels), lm)

    @torch.inference_mode()
    def _logits(self, audio):
        inputs = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        return self.model(**inputs.to(self.device)).logits[0]

    def transcribe_raw(self, audio, use_lm=True):
        texts = []
        for segment in split_long_audio(audio):
            logits = self._logits(segment)
            if self.decoder is not None and use_lm:
                logp = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
                text = self.decoder.decode(logp, beam_width=self.beam_width)
                text = text.replace(" ", "").replace("#", " ")
            else:
                text = self.processor.batch_decode(logits.argmax(-1)[None])[0]
            texts.append(text.replace("</s>", "").strip())
        return " ".join(t for t in texts if t)

    def transcribe(self, audio, use_lm=True):
        return clean_transcript(self.transcribe_raw(audio, use_lm))


WHISPER_ID = "openai/whisper-large-v3-turbo"
# Whisper's "prompt" primes the decoder with vocabulary/style (max ~224 tokens). Override by
# writing your own text to prompts/whisper_prompt.txt.
WHISPER_PROMPT_FILE = Path(__file__).parent / "prompts" / "whisper_prompt.txt"
DEFAULT_WHISPER_PROMPT = (
    "Cardiac catheterization and PCI report. Right radial access, 6 French JR 3.5 and EBU 3.5 "
    "guiding catheters. BMW and Sion Blue guidewires. Predilation with NC Trek and Mini Trek "
    "semi-compliant balloons at 12 atm. Xience Alpine everolimus-eluting stent 2.75 x 28 mm. "
    "In-stent restenosis of the RCA, LAD, LCx, PDA, PLB, OM1. TIMI grade 3 flow, well-apposed "
    "stent, no dissection. Dual antiplatelet therapy."
)


class WhisperASR:
    """Whisper (default large-v3-turbo) with the same transcribe() interface as MedASR.

    use_lm=True means "use the vocabulary prompt" -- Whisper's counterpart of the MedASR LM.
    """

    def __init__(self, model_id=WHISPER_ID, prompt=None, language="en", adapter=None):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # The checkpoint is fp16; CPUs need fp32 (fp16 conv/matmul are unsupported or very slow).
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.language = language
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=self.dtype)
        if adapter:  # LoRA fine-tune on top of the base model; merged once for plain-speed inference
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).merge_and_unload()
        self.model = self.model.to(self.device).eval()
        if prompt is None:
            prompt = (WHISPER_PROMPT_FILE.read_text(encoding="utf-8").strip()
                      if WHISPER_PROMPT_FILE.exists() else DEFAULT_WHISPER_PROMPT)
        self.prompt = prompt
        self.prompt_ids = (self.processor.get_prompt_ids(prompt, return_tensors="pt").to(self.device)
                           if prompt else None)

    @torch.inference_mode()
    def _transcribe_segment(self, audio, use_prompt):
        features = self.processor(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt").input_features
        kwargs = {"prompt_ids": self.prompt_ids} if use_prompt and self.prompt else {}
        ids = self.model.generate(features.to(self.device, self.dtype), language=self.language,
                                  task="transcribe", **kwargs)
        text = self.processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
        if use_prompt and text.startswith(self.prompt):  # older transformers echo the prompt
            text = text[len(self.prompt):].strip()
        return text

    def transcribe(self, audio, use_lm=True):
        return " ".join(t for t in (self._transcribe_segment(s, use_lm) for s in split_long_audio(audio)) if t)


class WhisperAPI:
    """Whisper large-v3-turbo via Hugging Face Inference Providers (cloud).

    AUDIO LEAVES THIS MACHINE. For quick accuracy previews on non-patient test audio only.
    The API has no prompt parameter, so there is no cardiology biasing (use_lm is ignored).
    Needs an HF token with the "Inference Providers" permission (`hf auth login`).
    """

    device = "cloud"

    def __init__(self, model_id=WHISPER_ID, provider="auto"):
        from huggingface_hub import InferenceClient

        self.model_id, self.client = model_id, InferenceClient(provider=provider)
        print("WARNING: whisper-api sends audio to an external cloud service. "
              "Do not use it with patient audio.", flush=True)

    def transcribe(self, audio, use_lm=True):
        import tempfile

        import soundfile as sf

        texts = []
        for segment in split_long_audio(audio):
            # A .wav path (not raw bytes) so the client sends Content-Type: audio/wav.
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "segment.wav"
                sf.write(path, segment, SAMPLE_RATE, subtype="PCM_16")
                result = self.client.automatic_speech_recognition(str(path), model=self.model_id)
            texts.append(result.text.strip())
        return " ".join(t for t in texts if t)


def local_model_file(repo_id, filename):
    """Path of one already-downloaded model file (python download_model.py <repo_id>).

    Per-file lookup on purpose: download_model.py skips READMEs/sample wavs, and a whole-snapshot
    lookup would then report the model as incomplete.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        return hf_hub_download(repo_id, filename, local_files_only=True)
    except LocalEntryNotFoundError:
        raise SystemExit(f"{filename} of {repo_id} is not downloaded yet. Run:\n"
                         f"    python download_model.py {repo_id}") from None


PARAKEET_ID = "csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8"
HOTWORDS_FILE = Path(__file__).parent / "prompts" / "hotwords.txt"


class ParakeetASR:
    """NVIDIA Parakeet TDT 0.6B v2 (English), int8 ONNX via sherpa-onnx -- built for CPU.

    Outputs punctuation and capitalization itself. use_lm=True enables hotword boosting from
    prompts/hotwords.txt (one term per line) when that file exists.
    """

    def __init__(self, repo_id=PARAKEET_ID, num_threads=None):
        import sherpa_onnx
        self.device = "cpu"
        threads = num_threads or max(1, (os.cpu_count() or 2) // 2)  # physical cores
        common = dict(encoder=local_model_file(repo_id, "encoder.int8.onnx"),
                      decoder=local_model_file(repo_id, "decoder.int8.onnx"),
                      joiner=local_model_file(repo_id, "joiner.int8.onnx"),
                      tokens=local_model_file(repo_id, "tokens.txt"),
                      num_threads=threads, sample_rate=SAMPLE_RATE, feature_dim=80,
                      model_type="nemo_transducer")
        self.plain = sherpa_onnx.OfflineRecognizer.from_transducer(decoding_method="greedy_search", **common)
        self.boosted = None
        if HOTWORDS_FILE.exists():
            self.boosted = sherpa_onnx.OfflineRecognizer.from_transducer(
                decoding_method="modified_beam_search", hotwords_file=str(HOTWORDS_FILE),
                hotwords_score=1.5, **common)

    def transcribe(self, audio, use_lm=True):
        recognizer = self.boosted if use_lm and self.boosted else self.plain
        texts = []
        for segment in split_long_audio(audio):
            stream = recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment)
            recognizer.decode_stream(stream)
            texts.append(stream.result.text.strip())
        return " ".join(t for t in texts if t)


WHISPER_ONNX_SIZES = ("small", "medium", "turbo")


class WhisperOnnxASR:
    """Whisper (small / medium / turbo) as int8 ONNX via sherpa-onnx -- CPU-friendly Whisper.

    language="ur" makes Whisper write Urdu script; any English it keeps in English letters is
    Whisper's own choice (it has seen mixed Urdu-English text in training).
    Download: python download_model.py csukuangfj/sherpa-onnx-whisper-<size> --only "*int8.onnx,*tokens.txt"
    """

    def __init__(self, size="turbo", language="ur", num_threads=None):
        import sherpa_onnx

        repo_id = f"csukuangfj/sherpa-onnx-whisper-{size}"
        self.device = "cpu"
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=local_model_file(repo_id, f"{size}-encoder.int8.onnx"),
            decoder=local_model_file(repo_id, f"{size}-decoder.int8.onnx"),
            tokens=local_model_file(repo_id, f"{size}-tokens.txt"),
            language=language, task="transcribe",
            num_threads=num_threads or max(1, (os.cpu_count() or 2) // 2))

    def transcribe(self, audio, use_lm=True):
        texts = []
        for segment in split_long_audio(audio):  # Whisper's window is 30 s
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment)
            self.recognizer.decode_stream(stream)
            texts.append(stream.result.text.strip())
        return " ".join(t for t in texts if t)


DOLPHIN_IDS = {
    "dolphin-base": "csukuangfj/sherpa-onnx-dolphin-base-ctc-multi-lang-int8-2025-04-02",    # 140M, 104 MB
    "dolphin-small": "csukuangfj/sherpa-onnx-dolphin-small-ctc-multi-lang-int8-2025-04-02",  # 372M, 250 MB
}


class DolphinASR:
    """DataoceanAI Dolphin (40 Eastern languages incl. Urdu, Hindi), CTC, int8 ONNX via sherpa-onnx.

    Small and fast. Urdu output additionally gets English medical terms in English letters
    (english_terms.py); that step leaves non-Urdu text untouched.
    """

    def __init__(self, size="dolphin-base", num_threads=None):
        import sherpa_onnx

        repo_id = DOLPHIN_IDS[size]
        self.device = "cpu"
        self.english_terms = True
        self.recognizer = sherpa_onnx.OfflineRecognizer.from_dolphin_ctc(
            model=local_model_file(repo_id, "model.int8.onnx"), tokens=local_model_file(repo_id, "tokens.txt"),
            num_threads=num_threads or max(1, (os.cpu_count() or 2) // 2))

    def transcribe(self, audio, use_lm=True):
        texts = []
        for segment in split_long_audio(audio):
            stream = self.recognizer.create_stream()
            stream.accept_waveform(SAMPLE_RATE, segment)
            self.recognizer.decode_stream(stream)
            texts.append(stream.result.text.strip())
        text = " ".join(t for t in texts if t)
        if self.english_terms:
            from english_terms import apply_english_terms

            text = apply_english_terms(text)
        return text


OMNI_IDS = {
    "omni": "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-int8-2025-11-12",
    "omni-1b": "csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-1B-ctc-int8-2025-11-12",
    # Our Urdu fine-tune (train_omni_kaggle.py): writes Urdu script + English letters natively.
    "omni-ft": str(Path(__file__).parent / "models" / "omni-urdu-ft"),
}


# Urdu alphabet (Urdu forms of ک ی ہ, plus ٹ ڈ ڑ ں ھ ے ...), common diacritics, digits and Urdu
# punctuation. Deliberately excludes other Arabic-script letters (Arabic ي ك ه, Sindhi, Pashto).
URDU_CHARS = (set("ءآأؤئابپتٹثجچحخدڈذرڑزژسشصضطظعغفقکگلمنںوہھۂۃیےۓ") | set("ًٌٍَُِّْٰ")
              | set("۰۱۲۳۴۵۶۷۸۹0123456789") | {" ", "۔", "،", "؟"})



# Hesitation sounds dropped from the fine-tuned Omni output (English letters, and Urdu ہمم / اممم).
FILLER_WORDS = {"uh", "uhh", "um", "umm", "hmm", "hm", "ah", "eh", "er", "ہمم", "اممم"}


class OmnilingualASR:
    """Meta Omnilingual ASR CTC (1600+ languages incl. Urdu and English), int8 ONNX on CPU.

    The model has no language setting and writes whatever script it "hears" -- for Urdu speech
    that is mostly Hindi (Devanagari), since spoken Urdu and Hindi are nearly identical. With
    script="urdu", its (otherwise correct) recognition is re-spelled in Urdu script by
    devanagari_to_urdu.py, using the model's own scores to choose between same-sounding Urdu letters.
    Output is plain text without punctuation. Segments are kept at <= 25 s (model limit: 40 s).
    """

    BLANK_ID = 0  # CTC blank (<s>), as used by sherpa-onnx for this model

    def __init__(self, size="omni", script=None, num_threads=None):
        import onnxruntime as ort

        repo_id = OMNI_IDS[size]
        model_file = lambda name: (str(Path(repo_id) / name) if Path(repo_id).is_dir()
                                   else local_model_file(repo_id, name))
        self.device = "cpu"
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads or max(1, (os.cpu_count() or 2) // 2)
        self.session = ort.InferenceSession(model_file("model.int8.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        tokens_path = model_file("tokens.txt")
        self.tokens = [line.rsplit(" ", 1)[0] for line in
                       Path(tokens_path).read_text(encoding="utf-8").splitlines()]
        self.special = {"<s>", "<pad>", "</s>", "<unk>"}
        self.script = script
        self.english_terms = True  # Urdu mode: medical English terms in English letters (vocab/)
        if script:
            # Urdu letters (not space/digits), for picking "the Urdu version" of a character.
            letters = URDU_CHARS - set("۰۱۲۳۴۵۶۷۸۹0123456789 ")
            self.urdu_mask = np.array([t in letters for t in self.tokens])
            self.token_id = {t: i for i, t in enumerate(self.tokens)}
        # The fine-tune writes Urdu and English natively but, unsure of a rare Urdu word, may slip
        # single English letters into it (تلoوs). Keep every word in one script instead.
        self.one_script_per_word = size.startswith("omni-ft")
        if self.one_script_per_word:
            self.arabic_mask = np.array([len(t) == 1 and "\u0600" <= t <= "\u06ff" for t in self.tokens])
            self.latin_mask = np.array([len(t) == 1 and "a" <= t <= "z" for t in self.tokens])

    def _decode_segment(self, audio):
        x = (audio - audio.mean()) / (audio.std() + 1e-7)  # same normalisation as sherpa-onnx
        logits = self.session.run(None, {"x": x[None].astype(np.float32)})[0][0]
        # Greedy CTC on the model's own choices: collapse repeats, drop blanks. The model decides
        # WHERE characters and word gaps are; script conversion only changes HOW they're spelled.
        seq, prev = [], None
        for frame, idx in enumerate(logits.argmax(-1)):
            if idx != prev and idx != self.BLANK_ID and self.tokens[idx] not in self.special:
                seq.append((self.tokens[idx], frame))
            prev = idx
        if self.script == "urdu":
            from devanagari_to_urdu import to_urdu

            masked = np.where(self.urdu_mask, logits, -np.inf)
            text = to_urdu(seq,
                           best_urdu=lambda f: self.tokens[int(masked[f].argmax())],
                           score=lambda f, ch: logits[f, self.token_id[ch]] if ch in self.token_id else -np.inf)
            if self.english_terms:
                from english_terms import apply_english_terms

                text = apply_english_terms(text)
        elif self.one_script_per_word:
            text = self._one_script_per_word(seq, logits)
            # The fine-tune learned to write hesitations ("uh", "um") from its training labels.
            text = " ".join(w for w in text.split() if w not in FILLER_WORDS)
        else:
            text = "".join(tok for tok, _ in seq)
        return " ".join(text.split())

    def _one_script_per_word(self, seq, logits):
        """Per word, letters of the minority script (Urdu vs English) are replaced by the model's
        best letter of the majority script at that frame; ties go to Urdu."""
        words, word = [], []
        for tok, frame in seq + [(" ", None)]:
            if tok.strip():
                word.append((tok, frame))
                continue
            if word:
                is_latin = [len(t) == 1 and "a" <= t <= "z" for t, _ in word]
                n_latin = sum(is_latin)
                n_arabic = sum("\u0600" <= t <= "\u06ff" for t, _ in word)
                if n_latin and n_arabic:
                    keep_latin = n_latin > n_arabic
                    mask = self.latin_mask if keep_latin else self.arabic_mask
                    fixed = []
                    for (t, f), lat in zip(word, is_latin):
                        if lat != keep_latin and (lat or "\u0600" <= t <= "\u06ff"):
                            t = self.tokens[int(np.where(mask, logits[f], -np.inf).argmax())]
                        fixed.append(t)
                    words.append("".join(fixed))
                else:
                    words.append("".join(t for t, _ in word))
            word = []
        return " ".join(words)

    def transcribe(self, audio, use_lm=True):
        return " ".join(t for t in (self._decode_segment(s) for s in split_long_audio(audio)) if t)


def load_asr(model="medasr", lm_path=None, **kwargs):
    """Factory used by the CLI scripts: 'medasr' (optionally with LM), 'parakeet', 'omni',
    'omni-1b', 'omni-urdu', 'omni-1b-urdu', 'omni-ft', 'whisper', 'whisper-urdu-base' or 'whisper-api'."""
    if model == "parakeet":
        return ParakeetASR()
    if model in DOLPHIN_IDS:
        return DolphinASR(model)
    if model == "whisper-urdu-base":
        return WhisperASR(WHISPER_ID, prompt="", language="ur")
    if model.startswith("whisper-") and model.endswith("-urdu"):  # whisper-turbo-urdu, ...
        return WhisperOnnxASR(model.split("-")[1], language="ur")
    if model.startswith("omni"):
        script = "urdu" if model.endswith("-urdu") else None
        return OmnilingualASR(model.removesuffix("-urdu"), script=script)
    if model == "whisper":
        return WhisperASR()
    if model == "whisper-api":
        return WhisperAPI()
    return MedASR(lm_path=lm_path, **kwargs)
