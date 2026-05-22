# Medication Prediction on MIMIC-IV

Code for comparing four biomedical LLMs on ICU medication prediction, across three modes: zero-shot (no retrieval), RAG, and supervised fine-tuning. Targets are ATC-3 codes derived from MIMIC-IV prescriptions.

## Models

- BioGPT (`microsoft/biogpt`)
- BioMistral (`BioMistral/BioMistral-7B`)
- Meditron (`epfl-llm/meditron-7b`)
- PMC-LLaMA (`axiong/PMC_LLaMA_13B`)

## Modes

| Mode    | Script suffix | What it does                                                  |
|---------|---------------|---------------------------------------------------------------|
| No-RAG  | `*_norag.py`  | Zero-shot prompt, no retrieval, no candidate set in prompt    |
| RAG     | `*_rag.py`    | kNN retrieval over training cases, top-k injected into prompt |
| FT      | `*_ft.py`     | LoRA fine-tune on the same train split, then evaluate         |

So 4 models x 3 modes = 12 scripts.

## Data

Built from MIMIC-IV `prescriptions.csv`. NDC codes are mapped to ATC-3 via RxNorm (`/REST/ndcstatus` -> RXCUI -> ATC). A drug-name fallback is used for unresolved NDCs. Both lookups are cached as JSON next to the CSVs.

Split: 80/20 train/test, seed 42.

## Running

Paths inside each script point to the original cluster layout (`/workspace/LLM_research/treatRag/...`). Edit `CSV_DIR` / `OUT_DIR` near the top of each file before running. Each script is standalone:

```
python biogpt_norag.py
python meditron_rag.py
python pmcllama_ft.py
```

Outputs land in `OUT_DIR`: per-sample predictions (CSV) and an `evaluation_summary.json` with Jaccard, F1, precision, recall, and bootstrap CIs.

## Requirements

- Python 3.10+
- PyTorch (CUDA build, bf16 used in FT runs)
- transformers, peft, accelerate, datasets
- scikit-learn, scipy, pandas, numpy, tqdm, requests

PMC-LLaMA FT was run on a single A100 80GB; the 7B models fit on 40GB.

## Notes

- BioGPT context is 1024 tokens — prompts are truncated with headroom for generation.
- RAG retrieval uses sentence embeddings over the training cases; no external KB.
- The FT scripts share a prompt builder that equalizes the feature set seen at train and inference time.
