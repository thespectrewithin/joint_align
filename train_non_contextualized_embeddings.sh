#!/usr/bin/env bash

# Hyper-param
lang=fr
thredshold=95    # {70, 80, 90, 95}
lr=1        # {1, 10, 25, 50}
epoch=20    # {10, 20}

# Tool Path
MUSE_PATH=$PWD/tools/MUSE
RCSLS_PATH=$PWD/tools/fastText/alignment
JOINT_ALIGN_PATH=$PWD
LANG_PAIR_PATH=$PWD/word_embeddings/en_${lang} # Path to language pair vocab and embeddings

TRAIN_DICO_PATH=$MUSE_PATH/data/crosslingual/dictionaries/en-${lang}.0-5000.txt
TEST_DICO_PATH=$MUSE_PATH/data/crosslingual/dictionaries/en-${lang}.5000-6500.txt

JOINT_EMBED=$LANG_PAIR_PATH/fasttext.en-${lang}.word.joint.300
SRC_VOCAB=$LANG_PAIR_PATH/en.word.vocab
TGT_VOCAB=$LANG_PAIR_PATH/${lang}.word.vocab
JOINT_VOCAB=$LANG_PAIR_PATH/en.${lang}.word.vocab

SRC_ONLY_EMBED=$LANG_PAIR_PATH/en_only_embedding.${thredshold}
TGT_ONLY_EMBED=$LANG_PAIR_PATH/${lang}_only_embedding.${thredshold}
JOINT_ONLY_EMBED=$LANG_PAIR_PATH/joint_only_embedding.${thredshold}

OUTPUT_EMBED=$LANG_PAIR_PATH/joint_align_embedding

# wiki data

# fastText
# TODO: output vocab and joint embedding to LANG_PAIR_PATH


# Vocabulary Reallocation
python $JOINT_ALIGN_PATH/vocab_reallocation.py --threshold ${thredshold} --src_vocab $SRC_VOCAB --tgt_vocab $TGT_VOCAB --joint_vocab $JOINT_VOCAB --input_embedding $JOINT_EMBED --src_only_output_path $SRC_ONLY_EMBED --tgt_only_output_path $TGT_ONLY_EMBED --joint_only_output_path $JOINT_ONLY_EMBED

# Alignment Refinement using RCSLS, replace this line with other alignment methods if needed
python $RCSLS_PATH/align.py --lr ${lr} --niter ${epoch} --src_emb $SRC_ONLY_EMBED --tgt_emb $TGT_ONLY_EMBED --dico_train $TRAIN_DICO_PATH --dico_test $TEST_DICO_PATH --output $SRC_ONLY_EMBED.aligned

# Merge embeddings
python $JOINT_ALIGN_PATH/merge_embedding.py --input_embedding_1 $SRC_ONLY_EMBED.aligned --input_embedding_2 $TGT_ONLY_EMBED --input_embedding_3 $JOINT_ONLY_EMBED --output_path $OUTPUT_EMBED
