# Low-Rank Adaptation for Spanish-to-Quechua Translation

To read the paper and its results, please refer to ```Evaluating Low-Rank Adaptation for Multilingual Translation in Low-Resource Settings.pdf```

Researching the adaptation of LLMs to languages that are not part of their training set, using fine-tuning techniques.

## Overview

Fine-tuning Facebook's **NLLB-200-distilled-600M** model for Spanish-to-Quechua translation using **LoRA** (Low-Rank Adaptation) on the `somosnlp-hackathon-2022/spanish-to-quechua` dataset. This project demonstrates parameter-efficient fine-tuning (~2% trainable parameters) for low-resource language pairs.

## Core Workflow

### 1. Model Setup
- **Base Model**: `facebook/nllb-200-distilled-600M` (M2M100 architecture)
- **Language Pair**: Spanish (`spa_Latn`) → Quechua (`que_Latn`)
- **LoRA Configuration**:
  - `r=8`, `lora_alpha=16`, `lora_dropout=0.05`
  - Target modules: `q_proj` and `v_proj` only
  - Results in ~2% trainable parameters vs full model

### 2. Training
- **Batch Configuration**: Size 4 with 8-step gradient accumulation (simulates batch 32)
- **Training Parameters**: 3 epochs, learning rate 1e-4, weight decay 0.01
- **Preprocessing**: Max sequence length 64 tokens with padding
- **Checkpointing**: Saves every 1000 steps, retains 2 most recent checkpoints

### 3. Evaluation Metrics

**Primary Metrics:**
- **BLEU**: Standard bilingual evaluation understudy score
- **ChrF++**: Character n-gram F-score (more robust for morphologically rich languages)

**Additional Metrics:**
- Exact match accuracy
- Normalized Levenshtein distance
- Semantic similarity (using `paraphrase-multilingual-mpnet-base-v2`)
- Jaccard, Dice, Cosine, and Overlap coefficients

