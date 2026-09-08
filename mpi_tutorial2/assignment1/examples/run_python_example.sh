#!/bin/bash
# classroom / local demo: run + check the python example submission
cd "$(dirname "$0")/demo_python_submission"
python3 ../check_submission.py . ../demo_input.txt
