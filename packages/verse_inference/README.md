# VerseInference

Model loading, state caching, streaming generation.

Part of the [Verse](../../README.md) framework. Provides model loaders
(HuggingFace-compatible), recurrent state caches for Mamba/RWKV, samplers
(greedy / top-k / top-p / temperature), streaming generators, and an optional
OpenAI-compatible HTTP server (FastAPI) for serving models on pure CPU.
