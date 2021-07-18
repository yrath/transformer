# -*- coding: utf-8 -*-

from collections import defaultdict, Counter

from torchtext.data.utils import get_tokenizer


def get_most_common_pair(word_counter):
    pairs = defaultdict(int)
    for word, freq in word_counter.items():
        bytes = word.split()
        for first, second in zip(bytes[:-1], bytes[1:]):
            pairs[(first, second)] += freq

    most_common_pair = sorted(pairs, key=pairs.get)[-1]
    return most_common_pair


def byte_pair_encoding(dataset, *tokenizers, vocab_size=30000):
    word_counter = Counter()

    for data in dataset:
        for idx, tokenizer in enumerate(tokenizers):
            tokens = tokenizer(data[idx])

            tokens = [" ".join(word) + " </w>" for word in tokens]
            word_counter.update(tokens)

    # initialize bpe vocabulary with character vocabulary
    bpe_vocab = set()
    for word in word_counter:
        for char in word.split():
            bpe_vocab.add(char)

    while len(bpe_vocab) < vocab_size:
        most_common_pair = get_most_common_pair(word_counter)
        pair_byte = "".join(most_common_pair)

        bpe_vocab.add(pair_byte)

        new_word_counter = {}
        for word, freq in word_counter.items():
            word = word.replace(f"{most_common_pair[0]} {most_common_pair[1]}", pair_byte)
            new_word_counter[word] = freq
        word_counter = new_word_counter

    return bpe_vocab
