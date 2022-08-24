#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t seg-ctrl-v6-postprocessing "$SCRIPTPATH"
