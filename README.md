# Medication Prediction (ICU Patients)

Code for comparing four biomedical LLMs on ICU medication prediction, across three modes: zero-shot (no retrieval), RAG, and supervised fine-tuning. Targets are ATC-3 codes.

## Models

- BioGPT (`microsoft/biogpt`)
- BioMistral (`BioMistral/BioMistral-7B`)
- Meditron (`epfl-llm/meditron-7b`)
- PMC-LLaMA (`axiong/PMC_LLaMA_13B`)

So 4 models x 3 modes = 12 scripts.


## Requirements

- Python 3.10+
- PyTorch (CUDA build, bf16 used in FT runs)
- transformers, peft, accelerate, datasets
- scikit-learn, scipy, pandas, numpy, tqdm, requests
