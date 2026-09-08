#!/bin/bash
# Public demo of the checker (no submission needed):
#   python3 check.py demo_input.txt examples/demo_output.txt
cd "$(dirname "$0")/.."
python3 check.py demo_input.txt examples/demo_output.txt
