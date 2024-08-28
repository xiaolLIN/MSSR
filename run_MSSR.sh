#!/bin/bash

result_file='result.txt'

python run_model.py --gpu_id=1 --model=MSSR --dataset="yelp" \
           --ssl=1 --cl='idropwc' --tau=1 --cllmd=0.12 --aaplmd=9 --sim='dot' \
		       --n_layers=4 --n_heads=8  \
           --attribute_predictor='linear' --aap='wi_wc_bce' --aap_gate=1 \
           --ip_mode='gating'  \
           --train_batch_size=1024 --pooling_mode='mean' --config_files="configs/yelp_ac.yaml" \
           --attribute_hidden_size=[256] --ada_fuse=1 --fusion_type=gate --result_file=${result_file};


python run_model.py --gpu_id=1 --model=MSSR --dataset="Amazon_Toys_and_Games" \
           --ssl=1 --cl='idropwc' --tau=1 --cllmd=0.08 --aaplmd=3 --sim='dot' \
           --n_layers=3 --n_heads=4  \
           --attribute_predictor='linear' --aap='wi_wc_bce' --aap_gate=1 \
           --ip_mode='gating' \
           --train_batch_size=2048 --pooling_mode='mean' --config_files="configs/Amazon_Toys_and_Games_ac.yaml" \
           --attribute_hidden_size=[128] --ada_fuse=1 --fusion_type=gate --result_file=${result_file};


python run_model.py --gpu_id=1 --model=MSSR --dataset="Amazon_Beauty" \
           --ssl=1 --cl='idropwc' --tau=1 --cllmd=0.14 --aaplmd=4 --sim='dot' \
           --n_layers=3 --n_heads=2  \
           --attribute_predictor='linear' --aap='wi_wc_bce' --aap_gate=1 \
           --ip_mode='gating' \
           --train_batch_size=2048 --pooling_mode='mean' --config_files="configs/Amazon_Beauty_ac.yaml" \
           --attribute_hidden_size=[256] --ada_fuse=1 --fusion_type=gate --result_file=${result_file};


python run_model.py --gpu_id=1 --model=MSSR --dataset="Amazon_Sports_and_Outdoors"  \
           --ssl=1 --cl='idropwc' --tau=1 --cllmd=0.14 --aaplmd=2 --sim='dot' \
           --n_layers=2 --n_heads=8  \
           --attribute_predictor='linear' --aap='wi_wc_bce' --aap_gate=1 \
           --ip_mode='gating'  \
           --train_batch_size=1024 --pooling_mode='mean' --config_files="configs/Amazon_Sports_and_Outdoors_ac.yaml" \
           --attribute_hidden_size=[256] --ada_fuse=1 --fusion_type=gate --result_file=${result_file}
