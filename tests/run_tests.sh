#!/usr/bin/env bash

# we must add the module source path as want to test from the source directly (not a package yet)
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

tests=(
	test_reverse_ad.py
	test_forward_ad.py
)
# 
if [[ $# -gt 0 && ${1} == 'pytest'* ]]; then
	driver="${@}"
fi



# run the tests
${driver} ${tests[@]}