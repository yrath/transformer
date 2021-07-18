# -*- coding: utf-8 -*-

import os
import json

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import torchtext
from torchtext.data.utils import get_tokenizer

from transformer.model import Transformer
from transformer.bpe import byte_pair_encoding

dirname = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(dirname, ".data")
BATCH_SIZE = 16
N_EPOCHS = 100

class Collater(object):
    def __init__(self, vocab, de_tokenizer, en_tokenizer):
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

        self.tokenized_words = {}

        self.vocab = vocab

    def encode_item(self, item):
        # replace word by BPE encoding tokens, starting with the largest tokens
        def encode(src):
            tokens = []
            for key in self.vocab:
                if key in src:
                    break
            else:
                return [self.vocab["<unk>"]]

            while src:
                idx = src.find(key)
                if idx < 0:
                    tokens.extend(encode(src))
                    break
                if idx > 0:
                    tokens.extend(encode(src[:idx]))
                tokens.append(self.vocab[key])
                src = src[idx+len(key):]

            return tokens

        outp = []
        for word in item:
            word += "</w>"
            if word in self.tokenized_words:
                tokens = self.tokenized_words[word]
            elif word in self.vocab:
                tokens = [self.vocab[word]]
                self.tokenized_words[word] = tokens
            else:
                try:
                    tokens = encode(word)
                except:
                    import pdb; pdb.set_trace()

                self.tokenized_words[word] = tokens

            outp.extend(tokens)
        return outp

    def __call__(self, data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_item = de_item.strip().strip(".")
            en_item = en_item.strip().strip(".")

            de_tensor = torch.tensor(self.encode_item(self.de_tokenizer(de_item)))
            en_tensor = torch.tensor(self.encode_item(self.en_tokenizer(en_item)))
            de_batch.append(torch.cat([torch.tensor([self.vocab["<bos>"]]), de_tensor,
                torch.tensor([self.vocab["<eos>"]])], dim=0))
            en_batch.append(torch.cat([torch.tensor([self.vocab["<bos>"]]), en_tensor,
                torch.tensor([self.vocab["<eos>"]])], dim=0))

        de_batch = pad_sequence(de_batch, padding_value=self.vocab["<pad>"], batch_first=True)
        en_batch = pad_sequence(en_batch, padding_value=self.vocab["<pad>"], batch_first=True)

        return de_batch, en_batch


def create_masks(src, target, ignore_idx, device):
    seq_size = target.size()[1]
    subsequent_mask = torch.tril(torch.ones(seq_size, seq_size)).to(device)

    src_mask = src == ignore_idx
    src_mask = src_mask.unsqueeze(-2)
    target_mask = target == ignore_idx
    target_mask = target_mask.unsqueeze(-2) & (subsequent_mask == 0)

    return src_mask, target_mask


def translate(model, loader, vocab, device, max_len=200):
    reverse_vocab = {value: key for key, value in vocab.items()}

    def decode(token_list):
        return "".join(token_list).replace("</w>", " ")

    for src_sentence, target_sentence in loader:
        print("Sentence to translate:")
        print(decode([reverse_vocab[token_idx.item()] for token_idx in src_sentence[0]]))

        src_sentence = src_sentence.to(device)
        pred_sentence = torch.tensor([[vocab["<bos>"]]])
        for _ in range(max_len):
            seq_size = pred_sentence.size()[0]
            src_mask, target_mask = create_masks(src_sentence, pred_sentence, vocab["<pad>"], device)

            pred = model(src_sentence, pred_sentence.to(device), src_mask, target_mask)

            next_token = torch.argmax(F.softmax(pred[:,-1,:], dim=-1))
            pred_sentence = torch.cat([pred_sentence, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
            if next_token == vocab["<eos>"]:
                break

        print("Result:")
        print(decode([reverse_vocab[token_idx.item()] for token_idx in pred_sentence[0]]))
        print("Target:")
        print(decode([reverse_vocab[token_idx.item()] for token_idx in target_sentence[0]]))


def train_epoch(model, optimizer, trainloader, vocab, device):
    model.train()
    total_loss = 0
    for src, target in trainloader:
        optimizer.zero_grad()
        src, target = src.to(device), target.to(device)

        target_input = target[:, :-1]
        target_output = target[:, 1:]

        src_mask, target_mask = create_masks(src, target_input, vocab["<pad>"], device)

        pred = model(src, target_input, src_mask, target_mask)
        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_output.contiguous().view(-1),
            ignore_index=vocab["<pad>"])
        print(loss)

        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss / len(trainloader)


def main(rebuild_vocab=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    # Create byte pair encoding
    vocab_path = os.path.join(data_dir, "bpe_vocab.json")
    if os.path.exists(vocab_path) and not rebuild_vocab:
        with open(vocab_path, "r") as vocab_file:
            vocab = json.load(vocab_file)
    else:
        vocab_iter = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
        vocab = byte_pair_encoding(vocab_iter, de_tokenizer, en_tokenizer)
        with open(vocab_path, "w") as vocab_file:
            json.dump(vocab, vocab_file, indent=4)
    vocab.sort(key=len, reverse=True)
    vocab.extend(["<unk>", "<bos>", "<eos>", "<pad>"])
    vocab = OrderedDict([token, idx] for idx, token in enumerate(vocab))

    collater = Collater(vocab, de_tokenizer, en_tokenizer)

    model = Transformer(len(vocab))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)

    model.train()
    for i_epoch in range(N_EPOCHS):
        train_dataset = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
        trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collater)
        try:
            training_loss = train_epoch(model, optimizer, trainloader, vocab, device)
            print(f"Epoch {i_epoch}: {training_loss:.3f}")
        except KeyboardInterrupt:
            break

    model.eval()
    valid_dataset = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
    validloader = DataLoader(valid_dataset, batch_size=1, collate_fn=collater)
    translate(model, validloader, vocab, device)

if __name__ == "__main__":
    main()
