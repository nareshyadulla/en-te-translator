import torch
import sentencepiece as spm

from transformer import Transformer


DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def greedy_decode(model, tokenizer, sentence, max_length=64):
    model.eval()

    src_ids = tokenizer.encode(sentence)

    # Add BOS and EOS
    src_ids = [1] + src_ids + [2]

    # Padding
    if len(src_ids) < max_length:
        src_ids += [0] * (max_length - len(src_ids))
    else:
        src_ids = src_ids[:max_length]

    src_tensor = torch.tensor(
        [src_ids],
        dtype=torch.long
    ).to(DEVICE)

    tgt_ids = [1]  # BOS token

    for _ in range(max_length):

        tgt_tensor = torch.tensor(
            [tgt_ids],
            dtype=torch.long
        ).to(DEVICE)

        with torch.no_grad():
            output = model(src_tensor, tgt_tensor)

        next_token = output[:, -1, :].argmax(dim=-1).item()

        tgt_ids.append(next_token)

        # Stop at EOS
        if next_token == 2:
            break

    translated_text = tokenizer.decode(tgt_ids[1:-1])

    return translated_text


def load_model():
    tokenizer = spm.SentencePieceProcessor(
        model_file="en_te.model"
    )

    vocab_size = tokenizer.get_piece_size()
    model = Transformer(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=192,
            num_heads=4,
            num_layers=3,
            d_ff=512,
            max_seq_length=64,
            dropout=0.1
        ).to(DEVICE)
    # model = Transformer(
    #     src_vocab_size=vocab_size,
    #     tgt_vocab_size=vocab_size,
    #     d_model=256,
    #     num_heads=8,
    #     num_layers=4,
    #     d_ff=1024,
    #     max_seq_length=64,
    #     dropout=0.1
    # ).to(DEVICE)

    model.load_state_dict(
        torch.load(
            "en_te_transformer.pth",
            map_location=DEVICE
        )
    )

    model.eval()

    return model, tokenizer


def main():

    model, tokenizer = load_model()

    test_sentences = [
        "How are you?",
        "I am learning transformers.",
        "The weather is very good today.",
        "My name is Naresh.",
        "I love machine learning.",
        "Can you help me?",
        "This is a beautiful place.",
        "We are going to school.",
        "She is reading a book.",
        "The food tastes delicious."
    ]

    print("\n==============================")
    print(" English → Telugu Translation ")
    print("==============================\n")

    for sentence in test_sentences:

        translation = greedy_decode(
            model,
            tokenizer,
            sentence
        )

        print(f"English : {sentence}")
        print(f"Telugu  : {translation}")
        print("-" * 50)


if __name__ == "__main__":
    main()