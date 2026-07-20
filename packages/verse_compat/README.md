# VerseCompat

HuggingFace / PyTorch compatibility adapters (optional, used only when installed).

Part of the [Verse](../../README.md) framework. Provides `load_hf_state_dict`
for reading `.bin` (PyTorch pickle) and `.safetensors` weights into
`verse_torch.Tensor` without requiring PyTorch at runtime, plus `torch_api`
aliases that map `torch.nn.Linear` etc. to VerseTorch equivalents to ease
porting of existing model code.
