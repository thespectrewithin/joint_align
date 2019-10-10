#!/usr/bin/env bash

# Hyper-param
src_lang=en
tgt_lang=fr
thredshold=90    # {70, 80, 90, 95}

# Tool Path
MUSE_PATH=$PWD/tools/MUSE
LANG_PAIR_PATH=$PWD/word_embeddings/${src_lang}_${tgt_lang} # Path to language pair vocab and embeddings

JOINT_ALIGN_EMBED=$LANG_PAIR_PATH/joint_align_embedding
SRC_VOCAB=$LANG_PAIR_PATH/${src_lang}.word.vocab
TGT_VOCAB=$LANG_PAIR_PATH/${tgt_lang}.word.vocab
JOINT_VOCAB=$LANG_PAIR_PATH/${src_lang}.${tgt_lang}.word.vocab

SRC_OUTPUT_EMBED=$LANG_PAIR_PATH/${src_lang}_embedding.${thredshold}
TGT_OUTPUT_EMBED=$LANG_PAIR_PATH/${tgt_lang}_embedding.${thredshold}

# Split Embeddings for Evaluation
python vocab_reallocation.py --threshold ${thredshold} --src_vocab $SRC_VOCAB --tgt_vocab $TGT_VOCAB --joint_vocab $JOINT_VOCAB --input_embedding $JOINT_ALIGN_EMBED --src_only_output_path $SRC_OUTPUT_EMBED --tgt_only_output_path $TGT_OUTPUT_EMBED

# Evaluate
python $MUSE_PATH/evaluate.py --src_lang ${src_lang} --tgt_lang ${tgt_lang} --src_emb $SRC_OUTPUT_EMBED --tgt_emb $TGT_OUTPUT_EMBED --max_vocab -1
