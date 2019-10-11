#!/usr/bin/env bash

# Monolingual Corpora
src_corpus=en.wikipedia.tok
tgt_corpus=fr.wikipedia.tok
concat_corpus=en-fr.wikipedia.tok

# Hyper-param
src_lang=en
tgt_lang=fr
topk=200000
thredshold=90    # {70, 80, 90, 95}
lr=10        # {1, 10, 25, 50}
epoch=10    # {10, 20}

# Tool Path
MUSE_PATH=$PWD/tools/MUSE
RCSLS_PATH=$PWD/tools/fastText/alignment
FASTTEXT=$PWD/tools/fastText/fasttext
FASTBPE=$PWD/tools/fastBPE/fast
LANG_PAIR_PATH=$PWD/word_embeddings/${src_lang}_${tgt_lang} # Path to language pair vocab and embeddings

TRAIN_DICO_PATH=$MUSE_PATH/data/crosslingual/dictionaries/${src_lang}-${tgt_lang}.0-5000.txt
TEST_DICO_PATH=$MUSE_PATH/data/crosslingual/dictionaries/${src_lang}-${tgt_lang}.5000-6500.txt

JOINT_EMBED=$LANG_PAIR_PATH/fasttext.${src_lang}-${tgt_lang}.word.joint.300
SRC_VOCAB=$LANG_PAIR_PATH/${src_lang}.word.vocab
TGT_VOCAB=$LANG_PAIR_PATH/${tgt_lang}.word.vocab
JOINT_VOCAB=$LANG_PAIR_PATH/${src_lang}.${tgt_lang}.word.vocab

SRC_ONLY_EMBED=$LANG_PAIR_PATH/${src_lang}_only_embedding.${thredshold}
TGT_ONLY_EMBED=$LANG_PAIR_PATH/${tgt_lang}_only_embedding.${thredshold}
JOINT_ONLY_EMBED=$LANG_PAIR_PATH/joint_only_embedding.${thredshold}

OUTPUT_EMBED=$LANG_PAIR_PATH/joint_align_embedding

mkdir $LANG_PAIR_PATH

# down (or up) sample a side of the corpus and concatenate them
a=($(wc -l $tgt_corpus))
python sample_corpus --corpus $src_corpus --target_size ${a[0]} --output $src_corpus.sampled
cat $src_corpus.sampled $tgt_corpus | shuf > $concat_corpus

# train joint fastText embeddings
$FASTTEXT skipgram -dim 300 -thread 24 -input $concat_corpus -output $concat_corpus.300

# get the vocab and counts for the next steps
$FASTBPE getvocab $src_corpus.sampled > $SRC_VOCAB
$FASTBPE getvocab $tgt_corpus > $TGT_VOCAB
$FASTBPE getvocab $src_corpus.sampled $tgt_corpus > $JOINT_VOCAB

# Select topk words for
python topk_embedding.py --src_vocab $SRC_VOCAB --tgt_vocab $TGT_VOCAB --input_embedding $JOINT_EMBED  --output_path $JOINT_EMBED.$topk --topk $topk --src_lang ${src_lang} --tgt_lang ${tgt_lang}  --dico_path $MUSE_PATH/data/crosslingual/dictionaries

# Vocabulary Reallocation
python vocab_reallocation.py --threshold ${thredshold} --src_vocab $SRC_VOCAB --tgt_vocab $TGT_VOCAB --joint_vocab $JOINT_VOCAB --input_embedding $JOINT_EMBED.$topk --src_only_output_path $SRC_ONLY_EMBED --tgt_only_output_path $TGT_ONLY_EMBED --joint_only_output_path $JOINT_ONLY_EMBED

# Alignment Refinement using RCSLS, replace this line with other alignment methods if needed
python $RCSLS_PATH/align.py --lr ${lr} --niter ${epoch} --src_emb $SRC_ONLY_EMBED --tgt_emb $TGT_ONLY_EMBED --dico_train $TRAIN_DICO_PATH --dico_test $TEST_DICO_PATH --output $SRC_ONLY_EMBED.aligned

# Merge embeddings
python merge_embeddings.py --input_embedding_1 $SRC_ONLY_EMBED.aligned --input_embedding_2 $TGT_ONLY_EMBED --input_embedding_3 $JOINT_ONLY_EMBED --output_path $OUTPUT_EMBED
