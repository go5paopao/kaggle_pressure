#!/bin/bash

EXP_NAME=$1
echo "exp name: $EXP_NAME"
cd src/$EXP_NAME

git checkout -b feature/run_$EXP_NAME
poetry run python train.py

git add .
git commit -m "finish run $EXP_NAME"
git push origin HEAD
