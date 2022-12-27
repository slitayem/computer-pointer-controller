#!/bin/bash
set -ox pipefail

# usage: ./download_models <models-dir>

MODELS_FOLDER=$1
MODELS_LIST="models.txt"

while IFS= read -r MODEL_NAME; do
   # https://docs.openvino.ai/2022.1/omz_tools_downloader.html
   omz_downloader  --name ${MODEL_NAME} -o $MODELS_FOLDER
done < "$MODELS_LIST"
