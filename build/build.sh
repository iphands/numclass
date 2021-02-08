#!/bin/bash
set -e

docker build -t numclass .

docker run \
       --rm \
       -it \
       -v /home/iphands/prog/numclass/data:/numclass/data:ro \
       -v /home/iphands/prog/numclass/src/py/keras/results:/numclass/src/py/keras/results:rw \
       -v /home/iphands/prog/numclass/src/py/keras/lib:/numclass/src/py/keras/lib:ro \
       numclass

