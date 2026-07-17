# English → Telugu Transformer

This project trains a Transformer model for English to Telugu translation.

## Features

- Uses ONLY Input and Output fields from dataset
- SentencePiece tokenizer
- Encoder-Decoder Transformer
- Apple Silicon (MPS) support
- Greedy decoding inference
- Padding + batching
- Model checkpoint saving

## Install

```
conda activate en-te  
```

```bash
pip install -r requirements.txt
```

## Dataset

Place dataset folder:

```text
english_telugu_dataset/
```

## Run Training

```bash
python train.py
```

## Output

Model checkpoint:

```text
en_te_transformer.pth
```

Tokenizer files:

```text
en_te.model
en_te.vocab
```
