#!/bin/bash

rm -rf ../data/audio/wavs/$1
cp -r ../data/audio/flacs/$1 ../data/audio/wavs/
find ../data/audio/wavs/$1 -iname "*.flac" | xargs rm

for flacfile in `find ../data/audio/ -iname "*.flac"`
do
    wavfile="${flacfile/flacs/wavs}"
    sox $flacfile -r 16k "${wavfile%.*}.wav"
done
