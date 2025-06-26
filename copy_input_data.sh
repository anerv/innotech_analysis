#!/bin/bash

# Check if a path is provided as input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filepath>"
    exit 1
fi

provided_path="$1"

# Task 1: Copy and rename config.yml to config-model.yml in the current directory
cp "${provided_path}/config.yml" "./config-model.yml" || { echo "Failed to copy and rename config.yml"; exit 1; }

# Task 2: Copy files ending with '_otp_geo.parquet' to current_directory/data/input
cp "${provided_path}"/results/data/*_otp_geo.parquet data/input/ || { echo "No files matching '_otp_geo.parquet' found or failed to copy"; exit 1; }

echo "Input data copied successfully."
