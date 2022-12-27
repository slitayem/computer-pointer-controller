#!/bin/bash
set -o pipefail

# usage: source scripts/create_ven.sh

ENV_NAME=".mouse-pointer-controller"
python -m venv ${ENV_NAME}
. ${ENV_NAME}/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
