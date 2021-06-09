#!/bin/bash

export PATH="/home/sambit/Documents/Projects/slp_dissertation/f0_reaper/REAPER/build:$PATH"
f0Folder="data/f0/"
pmFolder="data/pm/"
audioFolder="data/audio/"
files=( $(ls $audioFolder | grep wav) )

for FILE in "${files[@]}";do
	name=$(echo "$FILE" | cut -f 1 -d '.')
	reaper -i $audioFolder$FILE -f $f0Folder$name".f0" -p $pmFolder$name".pm" -a
done
