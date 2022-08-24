#!/usr/bin/env bash

./build.sh

docker save seg-ctrl-v6-postprocessing | gzip -c > seg-ctrl-v6-postprocessing.tar.gz
