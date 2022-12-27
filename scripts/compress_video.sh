#!/bin/bash
set -o pipefail


NPUT_FILE=$1
OUTPUT_FILE=$2

ffmpeg -i $INPUT_FILE -vcodec h264 -acodec mp2 $OUTPUT_FILE