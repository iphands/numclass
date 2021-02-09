#!/bin/bash
find $2 -type f | parallel -j${1} bash ./scripts/validate_single.sh
