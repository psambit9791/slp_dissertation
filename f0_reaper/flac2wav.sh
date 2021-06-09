#!/bin/bash

for flacfile in `find data/audio/ -iname "*.flac"`
do
    wavfile="${flacfile/flacs/wavs}"
    sox $flacfile -r 16k "${wavfile%.*}.wav"
done
