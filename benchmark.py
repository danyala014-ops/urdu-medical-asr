"""Compare several models on the same clips and print one table.

Usage:
    python benchmark.py tests/urduspeech_codeswitched --models omni-urdu,whisper-small-urdu,whisper-turbo-urdu
    python benchmark.py tests/fleurs_urdu --models omni-urdu,omni-ft

Every <name>.wav needs a <name>.txt reference. Models are loaded one at a time.
Columns:
    WER / CER       whole transcript (Urdu + English), punctuation ignored, English lower-cased
    Urdu CER        Urdu-script characters only -- how well the Urdu itself is recognised
    English kept    of the English (Latin-letter) words in the references, the share the model
                    wrote in English letters -- the code-switching score
    RTF             compute time / audio length (0.2 = five times faster than real time)
Each model's transcripts are saved to results/<folder>__<model>.txt for inspection.
"""

print("starting up (loading libraries, ~10-20 s)...", flush=True)

import argparse
import re
import time
from collections import Counter
from pathlib import Path

import jiwer
import librosa

from asr_engine import SAMPLE_RATE, load_asr

RESULTS_DIR = Path(__file__).parent / "results"


def normalise(text):
    return " ".join(jiwer.RemovePunctuation()(text.lower()).split())


def urdu_only(text):
    return " ".join(re.findall(r"[؀-ۿ]+", normalise(text)))


def latin_words(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", normalise(text))


def run_model(name, clips):
    t0 = time.perf_counter()
    asr = load_asr(name)
    if hasattr(asr, "english_terms"):
        asr.english_terms = True  # references here write English in English letters
    load_s = time.perf_counter() - t0

    refs, hyps, audio_s, compute_s, lines = [], [], 0.0, 0.0, []
    for wav, ref in clips:
        audio, _ = librosa.load(wav, sr=SAMPLE_RATE, mono=True)
        t0 = time.perf_counter()
        hyp = asr.transcribe(audio)
        compute_s += time.perf_counter() - t0
        audio_s += len(audio) / SAMPLE_RATE
        refs.append(ref)
        hyps.append(hyp)
        lines.append(f"### {wav.name}\nREF: {ref}\nHYP: {hyp}\n")
        print(f"  [{len(hyps)}/{len(clips)}] {wav.name}", end="\r", flush=True)
    print(" " * 60, end="\r")

    ref_n, hyp_n = [normalise(r) for r in refs], [normalise(h) for h in hyps]
    ref_u, hyp_u = [urdu_only(r) for r in refs], [urdu_only(h) for h in hyps]
    kept = total = 0
    for r, h in zip(refs, hyps):
        want, got = Counter(latin_words(r)), Counter(latin_words(h))
        total += sum(want.values())
        kept += sum(min(c, got[w]) for w, c in want.items())
    pairs_u = [(r, h) for r, h in zip(ref_u, hyp_u) if r]
    return {
        "model": name,
        "WER": jiwer.wer(ref_n, hyp_n),
        "CER": jiwer.cer(ref_n, hyp_n),
        "Urdu CER": jiwer.cer(*map(list, zip(*pairs_u))) if pairs_u else float("nan"),
        "English kept": kept / total if total else float("nan"),
        "RTF": compute_s / audio_s,
        "load s": load_s,
    }, lines


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folder")
    parser.add_argument("--models", required=True, help="comma-separated model names (see medasr_test.py)")
    parser.add_argument("--limit", type=int, help="only the first N clips")
    args = parser.parse_args()

    folder = Path(args.folder)
    clips = [(w, w.with_suffix(".txt").read_text(encoding="utf-8-sig").strip())
             for w in sorted(folder.glob("*.wav")) if w.with_suffix(".txt").exists()]
    clips = clips[:args.limit] if args.limit else clips
    if not clips:
        raise SystemExit(f"no .wav + .txt pairs in {folder}")
    RESULTS_DIR.mkdir(exist_ok=True)

    rows = []
    for name in [m.strip() for m in args.models.split(",") if m.strip()]:
        print(f"\n=== {name}: {len(clips)} clips", flush=True)
        try:
            row, lines = run_model(name, clips)
        except SystemExit as e:  # e.g. model not downloaded -> message, continue with the rest
            print(f"  skipped: {e}")
            continue
        rows.append(row)
        (RESULTS_DIR / f"{folder.name}__{name}.txt").write_text("\n".join(lines), encoding="utf-8")
        print("  " + "  ".join(f"{k} {v:.1%}" if k not in ("model", "RTF", "load s") else
                              f"{k} {v:.2f}" if k != "model" else "" for k, v in row.items()).strip())

    print(f"\n{'model':24s} {'WER':>7s} {'CER':>7s} {'Urdu CER':>9s} {'English kept':>13s} {'RTF':>6s}")
    for r in sorted(rows, key=lambda r: r["WER"]):
        print(f"{r['model']:24s} {r['WER']:7.1%} {r['CER']:7.1%} {r['Urdu CER']:9.1%} "
              f"{r['English kept']:13.1%} {r['RTF']:6.2f}")
    print(f"\ntranscripts: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
