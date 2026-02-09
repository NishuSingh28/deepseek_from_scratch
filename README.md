# DeepSeek-Style MoE Language Model — From Scratch

This repository implements a **DeepSeek-style Mixture-of-Experts (MoE) large language model** from first principles.

## Project Goals

- Decoder-only Transformer
- Sparse Mixture-of-Experts (MoE) FFN layers
- Token-level Top-k routing
- Load-balancing auxiliary loss
- Fully configurable for scale-up

## Repository Structure
```
deepseek_from_scratch/
├── configs/            # YAML configs for model & training
├── tokenizer/          # Tokenizer training and artifacts
├── model/              # Transformer, attention, MoE layers
├── routing/            # Router + load-balancing logic
├── data/               # Data generation & preprocessing
├── training/           # Losses, trainer, optimizers
├── inference/          # Generation & caching
├── utils/              # Logging & metrics
├── requirements.txt
└── README.md
```

## Current Status

- [ ] Environment setup
- [ ] Tokenizer
- [ ] Dense Transformer
- [ ] MoE FFN
- [ ] Routing stabilization
- [ ] Pretraining
