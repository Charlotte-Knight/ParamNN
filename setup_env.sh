#!/usr/bin/env bash

set -x
set -e

mkdir ${PWD}/tmp
export TMPDIR=${PWD}/tmp #often the default tmp dir is not big enough to install pytorch

python3 -m venv env
source env/bin/activate

pip install torch
pip install numpy
pip install matplotlib
pip install pandas
pip install sklearn

rm -r ${PWD}/tmp
