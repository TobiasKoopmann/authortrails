#!/usr/bin/env bash
# Usage, either "./run_local authortrails".py or "./run_local authortrails"
export PYSPARK_PYTHON=../../venv/bin/python3
# shellcheck disable=SC2155
export JAVA_HOME=$(/usr/libexec/java_home -v 1.8)
spark-submit authortrails.py "$1"