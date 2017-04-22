#!/bin/bash

# Set foler pathes for noisy and enhanced files
if [ $# -lt 2 ]; then
    echo 'ERROR: Noisy and Ehnahced Pathes should be provided!'
    echo "Usage: $0 <noisy_path> <enhanced_path>"
    exit 1
fi

NOISY_PATH="$1"
ENHANCED_PATH="$2"
mkdir -p $ENHANCED_PATH

FILES=$NOISY_PATH/*.wav

for f in $FILES 
do 
  clean_wav.sh $f $ENHANCED_PATH
done
