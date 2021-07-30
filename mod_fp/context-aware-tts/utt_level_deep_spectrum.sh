#!/bin/bash

WAVF=$1 # folder path
OUTF=$2 # .csv file

deepspectrum features $WAVF -nl -en vgg16 -fl fc2 -m mel -o $OUTF
