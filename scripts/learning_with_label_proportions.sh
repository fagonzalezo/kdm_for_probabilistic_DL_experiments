#!/bin/bash

git clone https://github.com/Z-Jianxin/Learning-from-Label-Proportions-A-Mutual-Contamination-Framework.git
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
wget https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test 
mkdir Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/data
mkdir Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/data/adult
mv adult.* Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/experiments/data/adult

cd Learning-from-Label-Proportions-A-Mutual-Contamination-Framework/LMMCM/
python make_data.py load_adult adult0 ./experiments/ ./experiments/ 0 0.5 8192 3000
python make_data.py load_adult adult1 ./experiments/ ./experiments/ 0.5 1.0 8192 3000
