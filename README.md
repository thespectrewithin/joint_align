# Joint_Align: A Unified Framework for Cross-lingual Alignment and Joint Training 
![Model](./illustration.png)

This repo contains the source codes for our paper

>[Cross-lingual Alignment vs Joint Training: A Comparative Study and A Simple Unified Framework]

>Zirui Wang*, Jiateng Xie*, Ruochen Xu, Yiming Yang, Graham Neubig, Jaime Carbonell (*: equal contribution)

>Preprint
>
Joint_Align is a unified framework for cross-lingual word embeddings (CLWE). The goal is to use unsupervised joint training as a coarse initialization and then applies alignment methods for refinement. Specifically, it contains three main components: (1) Joint Initialization (2) Vocabulary Reallocation (3) Alignment Refinement. Please see our paper for details.

This repo includes two settings where Joint_Align is applied to both non-contextualized and contextualized word embeddings. 



## Dependencies

* Python 3
* [NumPy](http://www.numpy.org/)
* [PyTorch](http://pytorch.org/)
* [fastText](https://github.com/facebookresearch/fastText) 
* [MUSE](https://github.com/facebookresearch/MUSE)
* [Moses](http://www.statmt.org/moses/)
* [fastBPE](https://github.com/glample/fastBPE)



## Joint_Align for Non-contextualized Word Embeddings
The script `train_non_contextualized_embeddings.sh` shows how to use this code to learn cross-lingual non-textualized word embeddings.

## Joint_Align for Contextualized Word Embeddings


## Application: Bilingual Lexicon Induction (BLI)
The script `example_BLI.sh` shows how to evaluate the cross-lingual non-textualized word embeddings learned on the BLI task using the MUSE benchmark dataset. Notice that it uses the official evaluation script of MUSE and the results correspond to Table 4 in our paper.

To reproduce results in Table 1, please use the following evaluation script (adapted from MUSE) which marks excluded test pairs as incorrect:
``` 
DICO_EVAL=/path/to/dico/en-${lang}.5000-6500.txt

python evaluate_BLI.py --src_emb $SRC_OUTPUT_EMBED --tgt_emb $TGT_OUTPUT_EMBED --dico_path $DICO_EVAL
```

For Russian, please use this [code](https://github.com/facebookresearch/XLM/blob/master/tools/lowercase_and_remove_accent.py) to remove accent from the dictionary.

## Application: Name Entity Recognition (NER)
