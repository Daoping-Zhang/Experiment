#!/bin/bash
# classroom / local demo: run + check the C example submission
cd "$(dirname "$0")/demo_c_submission"
python3 ../check_submission.py . ../demo_input.txt
