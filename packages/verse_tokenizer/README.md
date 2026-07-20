# VerseTokenizer

Lightweight BPE/Unigram tokenizer (no heavy deps).

Part of the [Verse](../../README.md) framework. Provides a minimal BPE/Unigram
tokenizer that can run without the `tokenizers` or `sentencepiece` runtime
dependencies, while still being able to load HuggingFace `tokenizer.json`
files for compatibility.
