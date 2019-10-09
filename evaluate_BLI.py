
import argparse
import torch

from utils import *

# main
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", type=bool, default=True, help="Run on GPU")
# data
parser.add_argument("--dico_path", type=str, default="", help="Test Dictionary")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default="", help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default="", help="Reload target embeddings")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")


# parse parameters
params = parser.parse_args()

# load data
print("Testing src emb:", params.src_emb)
print("Testing tgt emb:", params.tgt_emb)
src_word2id, src_emb, src_words = load_embedding(params.src_emb, params.max_vocab)
tgt_word2id, tgt_emb, tgt_words = load_embedding(params.tgt_emb, params.max_vocab)

src_emb = torch.from_numpy(src_emb).float()
tgt_emb = torch.from_numpy(tgt_emb).float()
if params.cuda:
    src_emb = src_emb.cuda()
    tgt_emb = tgt_emb.cuda()

precision = get_word_translation_accuracy(src_word2id, src_emb, tgt_word2id, tgt_emb, params.dico_path)

print("Testing on dictionary:", params.dico_path)
print("Precision @ 1:", "%.1f" % precision)
