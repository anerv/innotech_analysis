#!/bin/bash

# Check if a path is provided as input
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filepath>"
    exit 1
fi

provided_path="$1"

cp "${provided_path}/config.yml" "./config_model.yml" || { echo "Failed to copy and rename config.yml"; exit 1; }

cp "${provided_path}"/results/data/*_otp_geo.parquet data/input/ || { echo "No files matching '_otp_geo.parquet' found or failed to copy"; exit 1; }

cp "${provided_path}"/data/processed/adm_boundaries/study_area.gpkg data/input/ || { echo "Study area gpkg not found or failed to copy"; exit 1; }

cp "${provided_path}"/data/processed/adm_boundaries/municipalities.parquet data/input/ || { echo "Municipalities file not found or failed to copy"; exit 1; }



echo "Input data copied successfully."
