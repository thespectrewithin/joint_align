#!/bin/bash

set -e

TOOLS=$CUR/tools
mkdir -p $TOOLS

cd $TOOLS

#fastText
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip install .
cd $TOOLS

#MUSE
git clone https://github.com/facebookresearch/MUSE
cd MUSE/data
./get_evaluation.sh
cd $TOOLS

#fast_align
git clone https://github.com/clab/fast_align
cd fast_align
mkdir build
cd build
cmake ..
make
cd $TOOLS

#fastBPE
git clone https://github.com/glample/fastBPE
cd fastBPE
g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
cd $TOOLS
