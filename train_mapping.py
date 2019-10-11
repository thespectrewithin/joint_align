#!/usr/bin/env python

"""Script to train mapping given aligned embeddings

"""

import os
import sys
import argparse
import time

import numpy as np

def main(arguments):

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--emb_file', help="Input embedding file affix")
    parser.add_argument('--output_file', help="Output model file path",
                        default='output')
    parser.add_argument('--orthogonal', action='store_true', \
                        help='use orthogonal constrained mapping')

    args = parser.parse_args(arguments)

    print(args)

    # read aligned embeddings
    suffix = ['.src', '.trg']
    embeds = [None, None]

    for j in [0, 1]:
        embeds[j] = np.loadtxt(args.emb_file+suffix[j], delimiter=' ')

    # NumPy/CuPy management
    x, z = embeds[0], embeds[1]
    print(x.shape, z.shape)
    print(z[1])
    xp = np

    # learn the mapping w
    # x.dot(w) \approx z
    if args.orthogonal:  # orthogonal mapping
        u, s, vt = xp.linalg.svd(z.T.dot(x))
        w = vt.T.dot(u.T)
    else:  # unconstrained mapping
        x_pseudoinv = xp.linalg.inv(x.T.dot(x)).dot(x.T)
        w = x_pseudoinv.dot(z)

    # save the learned mapping w
    np.savetxt(args.output_file+'.map', w, delimiter=' ', fmt='%0.6f')

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
