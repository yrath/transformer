# -*- coding: utf-8 -*-

import os

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F

import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from transformer.model import Transformer

dirname = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(dirname, ".data")
BATCH_SIZE = 16
N_EPOCHS = 100

class Collater(object):
    def __init__(self, de_vocab, en_vocab, de_tokenizer, en_tokenizer):
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.de_tokenizer = de_tokenizer
        self.en_tokenizer = en_tokenizer

        self.de_bos = de_vocab["<bos>"]
        self.de_eos = de_vocab["<eos>"]
        self.de_pad = de_vocab["<pad>"]

        self.en_bos = en_vocab["<bos>"]
        self.en_eos = en_vocab["<eos>"]
        self.en_pad = en_vocab["<pad>"]

    def __call__(self, data_batch):
        de_batch, en_batch = [], []
        for (de_item, en_item) in data_batch:
            de_tensor = torch.tensor(self.de_vocab(self.de_tokenizer(de_item)))
            en_tensor = torch.tensor(self.en_vocab(self.en_tokenizer(de_item)))
            de_batch.append(torch.cat([torch.tensor([self.de_bos]), de_tensor, torch.tensor([self.de_eos])], dim=0))
            en_batch.append(torch.cat([torch.tensor([self.en_bos]), en_tensor, torch.tensor([self.en_eos])], dim=0))

        de_batch = pad_sequence(de_batch, padding_value=self.de_pad, batch_first=True)
        en_batch = pad_sequence(en_batch, padding_value=self.en_pad, batch_first=True)

        return de_batch, en_batch


def train_epoch(model, optimizer, trainloader):
    model.train()
    total_loss = 0
    for src, target in trainloader:
        optimizer.zero_grad()

        target_input = target[:, :-1]
        target_output = target[:, 1:]
        pred = model(src, target_input)

        loss = F.cross_entropy(pred.view(-1, pred.size(-1)), target_output.contiguous().view(-1))
        print(loss)

        loss.backward()
        optimizer.step()
        total_loss += loss

    return total_loss / len(trainloader)


def main(rebuild_vocab=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    de_tokenizer = get_tokenizer("spacy", language="de_core_news_sm")
    en_tokenizer = get_tokenizer("spacy", language="en_core_web_sm")

    def yield_tokens(dataset, tokenizer, idx=0):
        for data in dataset:
            yield(tokenizer(data[idx]))


    src_vocab_file = os.path.join(data_dir, "src_vocab.pth")
    if os.path.exists(src_vocab_file) and not rebuild_vocab:
        de_vocab = torch.load(src_vocab_file)
    else:
        vocab_iter = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
        de_vocab = build_vocab_from_iterator(yield_tokens(vocab_iter, de_tokenizer, idx=0),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        de_vocab.set_default_index(de_vocab["<unk>"])

        torch.save(de_vocab, src_vocab_file)

    target_vocab_file = os.path.join(data_dir, "target_vocab.pth")
    if os.path.exists(target_vocab_file) and not rebuild_vocab:
        en_vocab = torch.load(target_vocab_file)
    else:
        vocab_iter = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
        en_vocab = build_vocab_from_iterator(yield_tokens(vocab_iter, en_tokenizer, idx=1),
            specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        en_vocab.set_default_index(en_vocab["<unk>"])#

        torch.save(en_vocab, target_vocab_file)

    collater = Collater(de_vocab, en_vocab, de_tokenizer, en_tokenizer)

    model = Transformer(len(de_vocab), len(en_vocab))
    optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    for i_epoch in range(N_EPOCHS):
        train_dataset = torchtext.datasets.IWSLT2017(root=data_dir, split="train", language_pair=("de", "en"))
        trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collater)

        training_loss = train_epoch(model, optimizer, trainloader)
        print(f"Epoch {i_epoch}: {training_loss:.3f}")

if __name__ == "__main__":
    main()
