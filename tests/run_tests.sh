#!/usr/bin/env bash
tests=(
	test_reverse_ad.py
	test_forward_ad.py
)
# we must add the module source path as want to test from the source directly (not a package yet)
export PYTHONPATH="$(pwd -P)/../src":${PYTHONPATH}

# run the tests
pytest -v ${tests[@]}