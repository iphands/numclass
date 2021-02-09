#!/bin/bash
model=$1
output="`python test.py $model 2>&1 | fgrep -e incorrect`"
echo "${model}: ${output}"
