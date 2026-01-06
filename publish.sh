#!/bin/bash
set -o errexit

# clean up
pip3 install --upgrade build twine
rm -rf dist

# build and upload
python3 -m build
twine upload dist/*
