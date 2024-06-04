#!/bin/bash

model_name=transmil
proj_name=hnsc_hpv

approach=drop

device=cuda
max_bag_size=24000

dataset=test

flip_perc=1
strategy="$flip_perc"%-of-all

model_dir_root=/path/to/results_directory
model_path=$model_dir_root/trained_models/$proj_name/"$model_name"

results_dir_root=$model_dir_root/patch_flipping/$proj_name
results_dir=$results_dir_root/"$model_name"_"$dataset"_"$approach"_flip_"$flip_perc"_"$SLURM_JOB_ID"
mkdir "$results_dir"


explanation_folder=path/to/explanation_scores
explanation_path=$explanation_folder/test_predictions.csv

python3 evaluation_patch_flipping.py \
\
--model-path=$model_path \
--results-dir="$results_dir" \
--sel-checkpoint=best \
--dataset=$dataset \
--explanation-types lrp perturbation_keep attention gi grad2 \
--explain-scores-path=$explanation_path \
--precomputed-heatmap-types lrp perturbation_keep attention gi grad2 \
--max-bag-size=$max_bag_size \
--strategy=$strategy \
--explained-rel=logit \
--lrp-params '{"gamma": 0, "eps": 1e-8, "no_bias": 1}' \
--approach=$approach \
--device=$device \
--flipping \
--baseline \
--morl-abs

