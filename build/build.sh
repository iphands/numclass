#!/bin/bash
set -e
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJ_ROOT="$( cd -- "$SCRIPT_DIR/.." &> /dev/null && pwd )"

(cd $SCRIPT_DIR && docker build -t numclass . )

docker run \
       --rm \
       -it \
       -v "$PROJ_ROOT/data:/numclass/data:ro" \
       -v "$PROJ_ROOT/src/py/keras/results:/numclass/src/py/keras/results:rw" \
       -v "$PROJ_ROOT/src/py/keras/lib:/numclass/src/py/keras/lib:ro" \
       numclass

