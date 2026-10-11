"""Download a Hugging Face model into the local cache with a clear progress readout.

Usage:
    python download_model.py                                 # openai/whisper-large-v3-turbo
    python download_model.py google/medasr
    python download_model.py csukuangfj/sherpa-onnx-whisper-turbo --only "*int8.onnx,*tokens.txt"

Shows overall percent, MB downloaded, speed and ETA (from network traffic, so other downloads
running at the same time inflate it slightly). Safe to re-run: finished files are skipped.
"""

import os
import sys
import threading
import time

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")  # we print our own single progress line

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

# PyTorch weights + config/tokenizer only; skip duplicate TF/Flax/ONNX/.bin copies.
ALLOW = ["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "*.onnx"]


def matches(name, patterns):
    from fnmatch import fnmatch
    return any(fnmatch(name.split("/")[-1], pattern) for pattern in patterns)


def cached_bytes(repo_id):
    blobs = os.path.join(HF_HUB_CACHE, "models--" + repo_id.replace("/", "--"), "blobs")
    total = 0
    if os.path.isdir(blobs):
        for entry in os.scandir(blobs):  # includes *.incomplete files still being written
            try:
                total += entry.stat().st_size
            except FileNotFoundError:
                pass
    return total


def received_bytes():
    """System-wide bytes received. Xet downloads buffer in memory and only write the file at
    the end, so disk size alone can't show progress; network counters can."""
    import psutil
    return psutil.net_io_counters().bytes_recv


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 60}m{seconds % 60:02d}s" if seconds >= 60 else f"{seconds}s"


def main():
    args = sys.argv[1:]
    patterns = ALLOW
    if "--only" in args:  # e.g. --only "*int8.onnx,*tokens.txt" to skip full-precision copies
        i = args.index("--only")
        patterns = [p.strip() for p in args[i + 1].split(",")]
        del args[i:i + 2]
    repo_id = args[0] if args else "openai/whisper-large-v3-turbo"
    info = HfApi().model_info(repo_id, files_metadata=True)
    files = [s for s in info.siblings if matches(s.rfilename, patterns)]
    total = sum(s.size or 0 for s in files)
    print(f"{repo_id}: {len(files)} files, {total / 1e6:,.0f} MB to fetch -> {HF_HUB_CACHE}", flush=True)

    start_disk, start_net = cached_bytes(repo_id), received_bytes()
    result = {}

    def worker():
        try:
            result["path"] = snapshot_download(repo_id, allow_patterns=patterns)
        except Exception as e:  # surfaced after the progress loop
            result["error"] = e

    thread = threading.Thread(target=worker, daemon=True)
    t0 = time.time()
    thread.start()
    last_print = 0
    while thread.is_alive():
        thread.join(timeout=1)
        now = time.time()
        if now - last_print < 5 and thread.is_alive():
            continue
        last_print = now
        fetched = max(cached_bytes(repo_id) - start_disk, received_bytes() - start_net)
        done = min(start_disk + fetched, total)
        speed = fetched / max(now - t0, 1e-6)
        eta = (total - done) / speed if speed > 0 else float("inf")
        pct = 100 * done / total if total else 100
        bar = "#" * int(pct // 5) + "-" * (20 - int(pct // 5))
        print(f"[{bar}] {pct:5.1f}%  {done / 1e6:7,.0f}/{total / 1e6:,.0f} MB  "
              f"{speed / 1e6:5.2f} MB/s  ETA {fmt_time(eta) if eta != float('inf') else '--'}",
              flush=True)

    if "error" in result:
        sys.exit(f"download failed: {result['error']}\nRe-run to resume.")
    print(f"done in {fmt_time(time.time() - t0)}: {result['path']}", flush=True)


if __name__ == "__main__":
    main()
