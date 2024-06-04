#!/bin/bash -l

base_dir=/path/to/base/dir


python3 toy_experiment.py \
\
--results-dir ${base_dir}/results/adjacent_smil/transmil \
--num-repetitions 30 \
--dataset-type adjacent_smil \
\
--num-numbers 10 \
--num-instances 30 \
--sampling hierarchical \
--features-type mnist_resnet18 \
--features-path ${base_dir}/data/mnist_resnet18 \
--num-bags-train 2000 \
--num-bags-val 500 \
--num-bags-test 1000 \
\
--model-type transmil \
--n-out-layers 0 \
--learning-rate 0.0001 \
--weight-decay 0.0 \
--num-epochs 200 \
--patience 40 \
--checkpoint best \
\
--explanation-methods random attention perturbation_keep grad2 gi lrp \
--evaluated-classes all_classes
