#!/bin/bash

DATA_PATH="../data/audio/wavs/$1/LibriSpeech/$1/"
LEXICON_PATH="../data/librispeech-lexicon.txt"
OUTPUT_PATH="../alignment/$1"

mkdir $OUTPUT_PATH
mfa validate $DATA_PATH $LEXICON_PATH
mfa align $DATA_PATH $LEXICON_PATH english $OUTPUT_PATH
