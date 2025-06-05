#!/bin/bash
# Unix shell script for Omni CLI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
python3 "$DIR/omni_cli.py" "$@" 