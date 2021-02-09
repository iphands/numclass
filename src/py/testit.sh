#!/bin/bash
set -e
rm -rf ../../data/t/*
cd ../../data/t

dd if=../emnist-letters-train-images-idx3-ubyte of=test1 count=100
dd if=../scrubbed-letters-idx3-ubyte           of=test2 count=100

hexdump -v test1 >1
hexdump -v test2 >2

meld 1 2
