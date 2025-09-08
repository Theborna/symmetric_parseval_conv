#!/bin/bash

source /home/borna/miniconda3/etc/profile.d/conda.sh
# Activate your environment, you have to create it first
conda activate borna

# Your job script goes below this line
clear
echo 'Installing required packages'
pip install -r requirements.txt
echo 'Finished install'
clear
echo 'starting run'
python visualize.py -d cuda
echo 'finished run'