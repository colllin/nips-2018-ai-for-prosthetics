#!/bin/bash

export HOME=/home/ubuntu
source ~/.profile
cd ~/nips-2018-ai-for-prosthetics

pipenv install jupyter

pipenv run jupyter nbconvert --to notebook --execute Rollout\ Distributed.ipynb --ExecutePreprocessor.timeout=-1

shutdown
