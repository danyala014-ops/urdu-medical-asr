# Test sets

Both sets are 30 short clips (`<id>.wav`) with a reference transcript (`<id>.txt`).
Run them with `benchmark.py`, for example:

    python benchmark.py tests/urduspeech_codeswitched --models omni-urdu,omni-ft

| Folder | What | Source and licence |
|---|---|---|
| `fleurs_urdu/` | Formal read Urdu, no English | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) (`ur_pk`), CC-BY-4.0 |
| `urduspeech_codeswitched/` | Everyday Urdu mixed with English words (vlogs, dramas, interviews) | [UrduSpeech](https://huggingface.co/datasets/ASLP-lab/UrduSpeech) hand-corrected benchmark, CC-BY-4.0 (Haq et al., 2026, arXiv:2605.17846) |

None of these clips were used to train the fine-tuned model. See `urduspeech_codeswitched/SOURCE.md`.
