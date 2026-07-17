import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_from_disk
import sentencepiece as spm
from transformer import Transformer
from tqdm import tqdm
import os


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class TranslationDataset(Dataset):
    def __init__(self, dataset, tokenizer, max_length=64):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def pad_sequence(self, ids):
        if len(ids) < self.max_length:
            ids += [0] * (self.max_length - len(ids))
        else:
            ids = ids[:self.max_length]
        return ids

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # ONLY USE INPUT AND OUTPUT
        src_text = sample["Input"]
        tgt_text = sample["Output"]

        src_ids = self.tokenizer.encode(src_text)
        tgt_ids = self.tokenizer.encode(tgt_text)

        # Add special tokens
        src_ids = [1] + src_ids + [2]
        tgt_ids = [1] + tgt_ids + [2]

        src_ids = self.pad_sequence(src_ids)
        tgt_ids = self.pad_sequence(tgt_ids)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long)
        )


def prepare_tokenizer(dataset_path, tokenizer_prefix="en_te", vocab_size=16000):
    if os.path.exists(f"{tokenizer_prefix}.model"):
        print("Tokenizer already exists.")
        return

    print("Preparing tokenizer training data...")

    dataset = load_from_disk(dataset_path)["train"]

    combined_file = "combined.txt"

    with open(combined_file, "w", encoding="utf-8") as f:
        for sample in dataset:
            f.write(sample["Input"] + "\n")
            f.write(sample["Output"] + "\n")

    print("Training SentencePiece tokenizer...")

    spm.SentencePieceTrainer.train(
        input=combined_file,
        model_prefix=tokenizer_prefix,
        vocab_size=vocab_size,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3
    )

    print("Tokenizer training complete.")


def greedy_decode(model, tokenizer, sentence, max_length=64):
    model.eval()

    src_ids = tokenizer.encode(sentence)
    src_ids = [1] + src_ids + [2]

    if len(src_ids) < max_length:
        src_ids += [0] * (max_length - len(src_ids))
    else:
        src_ids = src_ids[:max_length]

    src_tensor = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)

    tgt_ids = [1]

    for _ in range(max_length):
        tgt_tensor = torch.tensor([tgt_ids], dtype=torch.long).to(DEVICE)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        tgt_ids.append(next_token)

        if next_token == 2:
            break

    decoded = tokenizer.decode(tgt_ids[1:-1])

    return decoded


def main():
    dataset_path = "./english_telugu_dataset"

    print("Loading dataset...")
    dataset = load_from_disk(dataset_path)["train"]

    print(dataset[0])

    # Train tokenizer
    prepare_tokenizer(dataset_path)

    tokenizer = spm.SentencePieceProcessor(model_file="en_te.model")

    vocab_size = tokenizer.get_piece_size()

    print("Vocabulary Size:", vocab_size)

    train_dataset = TranslationDataset(dataset, tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    # SMALL MODEL FOR MACBOOK
    d_model = 192
    num_heads = 4
    num_layers = 3
    d_ff = 512
    # d_model = 256
    # num_heads = 8
    # num_layers = 4
    # d_ff = 1024
    max_seq_length = 64
    dropout = 0.1

    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.9, 0.98),
        eps=1e-9
    )

    epochs = 10

    print("Starting training...")

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(train_loader)

        for src_batch, tgt_batch in progress_bar:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)

            optimizer.zero_grad()

            output = model(
                src_batch,
                tgt_batch[:, :-1]
            )

            loss = criterion(
                output.contiguous().view(-1, vocab_size),
                tgt_batch[:, 1:].contiguous().view(-1)
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()

            progress_bar.set_description(
                f"Epoch {epoch+1} Loss {loss.item():.4f}"
            )

        avg_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch+1}/{epochs} Average Loss: {avg_loss:.4f}")

        torch.save(model.state_dict(), "en_te_transformer.pth")

    print("Training Complete.")

    # TEST TRANSLATION
    test_sentence = "How are you?"

    translation = greedy_decode(
        model,
        tokenizer,
        test_sentence
    )

    print("\nEnglish:", test_sentence)
    print("Telugu:", translation)


if __name__ == "__main__":
    main()
