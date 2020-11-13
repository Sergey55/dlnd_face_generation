#!/bin/bash
python train.py --max_epochs 50 --gpus=1

gcloud compute instances stop $HOSTNAME