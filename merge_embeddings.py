import argparse
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument("--input_embedding_1", type=str, required=True)
parser.add_argument("--input_embedding_2", type=str, required=True)
parser.add_argument("--input_embedding_3", type=str)
parser.add_argument("--output_path", type=str, required=True)

parser.add_argument("--vocab_size", type=int, default=400000)

params = parser.parse_args()

output_path = params.output_path
dict1, emb1, words1 = load_embedding(params.input_embedding_1, params.vocab_size)
dict2, emb2, words2 = load_embedding(params.input_embedding_2, params.vocab_size)

output_content = ""
included_dict = set()
for w in dict1:
    output_content += w + " " + ' '.join([str(v) for v in emb1[dict1[w]]]) + "\n"
    included_dict.add(w)

for w in dict2:
    if w not in included_dict:
        output_content += w + " " + ' '.join([str(v) for v in emb2[dict2[w]]]) + "\n"
        included_dict.add(w)

if params.input_embedding_3 is not None:
    dict3, emb3, words3 = load_embedding(params.input_embedding_3, params.vocab_size)

    for w in dict3:
        if w not in included_dict:
            output_content += w + " " + ' '.join([str(v) for v in emb3[dict3[w]]]) + "\n"
            included_dict.add(w)

assert len(emb1[0]) == len(emb2[0]) == len(emb3[0])
dim = len(emb1[0])
output_content = str(len(included_dict)) + " " + str(dim) + "\n" + output_content
with open(output_path, "w") as file:
    file.write(output_content)