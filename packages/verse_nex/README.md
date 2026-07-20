# VerseNex

Transformer-alternative architectures (Mamba-2, RWKV-7, Linear Attention, Hybrid).

Part of the [Verse](../../README.md) framework. Implements linear-complexity
sequence models — selective state-space (Mamba-2), RWKV-7 time/channel mixing,
RetNet-style linear attention, and SSM + sparse-attention hybrid blocks —
on top of VerseTorch, with O(1)-state recurrent inference and O(N) parallel training.
