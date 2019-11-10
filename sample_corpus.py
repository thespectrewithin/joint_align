import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--corpus", type=str, required=True,
                    help="The corpus to be up or down sampled")
parser.add_argument("--target_size", type=int, required=True)
parser.add_argument("--output", type=str, required=True)

args = parser.parse_args()

text = []

for line in open(args.corpus):
    if len(line.strip()) > 0:
        text.append(line)

perm = np.random.permutation(len(text))
output = open(args.output, 'w')

if len(text) > args.target_size:
    for i in range(args.target_size):
        output.write(text[perm[i]])
else:
    a = args.target_size // len(text)
    for i in range(a):
        for j in range(len(text)):
            output.write(text[j])
    for i in range(args.target_size - a * len(text)):
        output.write(text[perm[i]])
        
output.close()
