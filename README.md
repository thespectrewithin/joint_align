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
* [Moses](http://www.statmt.org/moses/)
* [fastBPE](https://github.com/glample/fastBPE)
* [fastText](https://github.com/facebookresearch/fastText) 


## Joint_Align for Non-contextualized Word Embeddings

## Joint_Align for Contextualized Word Embeddings

## Applications: Bilingual Lexicon Induction (BLI)

## Applications: Name Entity Recognition (NER)

