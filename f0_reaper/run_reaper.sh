#!/bin/bash

rm -rf data/f0/wavs/$1
cp -r data/audio/wavs/$1 data/f0/wavs/
find data/f0/wavs/ -iname "*.wav" | xargs rm

rm -rf data/pm/wavs/$1
cp -r data/audio/wavs/$1 data/pm/wavs/
find data/pm/wavs/ -iname "*.wav" | xargs rm

export PATH="/home/sambit/Documents/Projects/slp_dissertation/f0_reaper/REAPER/build:$PATH"
audioFolder="data/audio/wavs/$1"

for FILE in `find $audioFolder -iname "*.wav"`;
do
	name=$(echo "$FILE" | cut -f 1 -d '.')
	reaper -i $FILE -f ${name/audio/f0}".f0" -p ${name/audio/pm}".pm" -a
done
