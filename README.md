# English → Telugu Transformer

A from-scratch encoder–decoder Transformer for English-to-Telugu machine translation — built as a hands-on learning project to understand how modern translation models work under the hood.

## What I'm Doing Here

This repo is my end-to-end experiment in building a neural machine translation system without relying on high-level libraries like Hugging Face Transformers or fairseq. The goal is not to beat production MT systems, but to **learn by implementing**.

The pipeline:

1. **Dataset** — English–Telugu sentence pairs from [MRR24/English_to_Telugu_Bilingual_Sentence_Pairs](https://huggingface.co/datasets/MRR24/English_to_Telugu_Bilingual_Sentence_Pairs), using only the `Input` and `Output` fields.
2. **Tokenization** — A shared SentencePiece tokenizer (16k vocab) trained on both languages.
3. **Model** — A custom Transformer (`transformer.py`) with multi-head attention, positional encoding, encoder/decoder stacks, and padding/causal masks.
4. **Training** — Teacher-forced seq2seq training on Apple Silicon (MPS), with a small model sized for a MacBook.
5. **Inference** — Greedy autoregressive decoding via `test.py`.

```
english_telugu_dataset/  →  SentencePiece  →  Transformer  →  models/v1/
                                                      ↓
                                              greedy decode (test.py)
```

### Project Structure

| File | Purpose |
|------|---------|
| `transformer.py` | Transformer architecture (attention, encoder, decoder) |
| `train.py` | Tokenizer training, dataset loading, training loop |
| `test.py` | Load checkpoint and run translations |
| `dataset_util.py` | Download/load the Hugging Face dataset locally |

## What I've Learned

- **Transformer internals** — How scaled dot-product attention, multi-head attention, layer norm, residual connections, and feed-forward blocks fit together in an encoder–decoder stack.
- **Masking matters** — Padding masks (ignore `<pad>`) and causal masks (prevent the decoder from peeking at future tokens) are essential for correct training.
- **Teacher forcing** — Feeding the decoder ground-truth tokens (`tgt[:, :-1]`) during training, while predicting the next token (`tgt[:, 1:]`), is the standard seq2seq training pattern.
- **Subword tokenization** — SentencePiece handles out-of-vocabulary words and morphologically rich languages like Telugu better than word-level tokenization.
- **Shared vocabulary** — Using one tokenizer for both source and target simplifies the embedding layer, though separate tokenizers can help for very different scripts.
- **Hardware-aware training** — Tuning `d_model`, layers, and batch size to fit Apple MPS memory (currently `d_model=192`, 3 layers, 4 heads).
- **End-to-end ML workflow** — Dataset management → tokenization → model definition → training loop → checkpoint saving → inference.

## What I Can Improve

### Model & Training
- [ ] Add a **validation split** and track loss/BLEU per epoch instead of training on all data blindly.
- [ ] Replace **greedy decoding** with **beam search** for better translation quality.
- [ ] Add **learning rate warmup + decay** (Noam schedule) instead of a fixed LR.
- [ ] Try **label smoothing** to reduce overconfidence on training tokens.
- [ ] Scale up the model (`d_model=256`, 4 layers, 8 heads) when more compute is available — configs are already commented in `train.py`.
- [ ] Train for more epochs with **early stopping** based on validation metrics.

### Data & Tokenization
- [ ] Use **separate tokenizers** for English and Telugu instead of a shared one.
- [ ] Filter overly long sentence pairs and add basic **data cleaning**.
- [ ] Add a held-out **test set** with quantitative evaluation (BLEU, chrF).

### Engineering
- [ ] Centralize hyperparameters in a **config file** instead of hardcoding in `train.py` and `test.py`.
- [ ] Wire `dataset_util.py` into the training script for a cleaner download/load flow.
- [ ] Add **Git LFS** for large checkpoints and dataset files.
- [ ] Save **versioned checkpoints** (e.g. `models/v2/`) with training metadata.
- [ ] Add a simple **CLI or web demo** for interactive translation.

## Install

```bash
conda activate en-te
pip install -r requirements.txt
```

## Dataset

Download and save locally (or place an existing copy):

```bash
python dataset_util.py
```

Expected path:

```text
english_telugu_dataset/
```

## Run Training

```bash
python train.py
```

## Run Inference

```bash
python test.py
```

## Output

### Artifacts

| Artifact | Description |
|----------|-------------|
| `models/v1/en_te_transformer.pth` | Model weights |
| `models/v1/en_te.model` / `en_te.vocab` | SentencePiece tokenizer |

### Sample translations (`python test.py`)

```text
==============================
 English → Telugu Translation 
==============================

English : How are you?
Telugu  : మీరు ఎలా ఉన్నారు?
--------------------------------------------------
English : I am learning transformers.
Telugu  : నేను  ⁇ స్స్స్థుతుతను.
--------------------------------------------------
English : The weather is very good today.
Telugu  : ఈ రోజు వాతావరణం చాలా బాగుంది.
--------------------------------------------------
English : My name is Naresh.
Telugu  : నా పేరు సుతుకుడు.
--------------------------------------------------
English : I love machine learning.
Telugu  : నాకు యంత్రం నేర్చుకోవడం చాలా ఇష్టం.
--------------------------------------------------
English : Can you help me?
Telugu  : మీరు నాకు సహాయం చేయగలరా?
--------------------------------------------------
English : This is a beautiful place.
Telugu  : ఇది అందమైన ప్రదేశం.
--------------------------------------------------
English : We are going to school.
Telugu  : మేము పాఠశాలకు వెళ్తున్నాము.
--------------------------------------------------
English : She is reading a book.
Telugu  : ఆమె ఒక పుస్తకం చదవడం.
--------------------------------------------------
English : The food tastes delicious.
Telugu  : ఆహారం రుచిగా ఉంటుంది.
--------------------------------------------------
```
