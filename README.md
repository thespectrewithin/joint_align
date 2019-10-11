# Joint_Align: A Unified Framework for Cross-lingual Alignment and Joint Training 
![Model](./illustration.png)

This repo contains the source codes for our paper

>[Cross-lingual Alignment vs Joint Training: A Comparative Study and A Simple Unified Framework](https://arxiv.org/abs/1910.04708)

>Zirui Wang*, Jiateng Xie*, Ruochen Xu, Yiming Yang, Graham Neubig, Jaime Carbonell (*: equal contribution)

>Preprint

## Introduction

Joint_Align is a unified framework for cross-lingual word embeddings (CLWE). The goal is to use unsupervised joint training as a coarse initialization and then applies alignment methods for refinement. Specifically, it contains three main components: (1) Joint Initialization (2) Vocabulary Reallocation (3) Alignment Refinement. Please see our paper for details.

This repo includes two settings where Joint_Align is applied to both non-contextualized and contextualized word embeddings. For non-contextualized embeddings, we show how to obtain one from scratch, and provide scripts to evaluate it on 2 downstream tasks, BLI and cross-lingual NER. For contextualized embeddings, we provide an example on how to apply our framework on [Multilingual BERT](https://github.com/google-research/bert/blob/master/multilingual.md), and evaluate it on cross-lingual NER.

## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [fastText](https://github.com/facebookresearch/fastText) 
* [MUSE](https://github.com/facebookresearch/MUSE)
* [fast_align](https://github.com/clab/fast_align)
* [fastBPE](https://github.com/glample/fastBPE)

To get started, run `./get_tools.sh`. 

## I. Non-contextualized Word Embeddings

### Train embeddings

First, we assume access to monolingual corpus such as Wikipedia for both languages. Use scripts such as [this one](https://github.com/facebookresearch/XLM/blob/master/get-data-wiki.sh) for getting the corpus.
The script `train_non_contextualized_embeddings.sh` shows how to use this code to learn cross-lingual non-textualized word embeddings. 
This will produce a joint_align embedding at the location `$PWD/word_embeddings/${src_lang}_${tgt_lang}/joint_align_embedding`, which can then be applied to downstream tasks.

### Application: Bilingual Lexicon Induction (BLI)

The script `example_BLI.sh` shows how to evaluate the cross-lingual non-textualized word embeddings learned on the BLI task using the MUSE benchmark dataset. Notice that it uses the official evaluation script of MUSE and the results correspond to Table 4 in our paper.

To reproduce results in Table 1, please use the following evaluation script (adapted from MUSE) which marks excluded test pairs as incorrect:
``` 
DICO_EVAL=/path/to/dico/${src_lang}-${tgt_lang}.5000-6500.txt

python evaluate_BLI.py --src_emb $SRC_OUTPUT_EMBED --tgt_emb $TGT_OUTPUT_EMBED --dico_path $DICO_EVAL
```

For Russian, please use this [code](https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py) to remove accent from the dictionary.

## II. Contextualized Word Embeddings

Joint_Align can be applied to Multilingual BERT by aligning its extracted features before feeding them to downstream models.

### Learn Alignment Matrix

First, we apply word alignment tools such as [fast_align](https://github.com/clab/fast_align) on parallel data, and learn alignment matrices using the features corresponding to the aligned words. To do so, simply run `./get_mapping.sh`.

### Application: Cross-lingual NER

After we obtain the alignment matrices, we can use them to align extracted features and feed these features for downstream tasks. The steps can be found in `run_feature_ner.sh`.

