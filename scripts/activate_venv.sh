#!/bin/bash
set -o pipefail

# usage: source scripts/activate_ven.sh

ENV_NAME=".mouse-pointer-controller"
. ${ENV_NAME}/bin/activate && echo $VIRTUAL_ENV