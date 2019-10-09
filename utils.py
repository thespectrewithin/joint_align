import numpy as np


def read_vocab(path):
    vocab, total_count = {}, 0
    for line in open(path, encoding="utf-8"):
        l = line.strip().split()
        if len(l) == 2:
            vocab[l[0]] = int(l[1])
            total_count += int(l[1])
    return vocab, total_count

def load_embedding(path, vocab_size):
    word_vector = []
    word_dict = {}
    words = []

    for line in open(path):
        if len(line.rstrip().split()) > 2:
            if len(words) < vocab_size:
                word, vec = line.rstrip().split(' ', 1)
                word_dict[word] = len(word_dict)
                words.append(word)
                vec = np.array(vec.split(), dtype='float32')
                word_vector.append(vec)

    return word_dict, np.vstack(word_vector), words