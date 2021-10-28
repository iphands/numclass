#!/bin/bash

brew install python-tk@3.9
cd src/py
python3 -mvenv venv
source venv/bin/activate
pip install -r requirements.txt
python ./keras/test_gui.py ./keras/model.hdf5
