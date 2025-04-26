#!/bin/bash
set -euo pipefail

# Download the London bike sharing dataset
curl -L -o ./london-bike-sharing-dataset.zip \
    https://www.kaggle.com/api/v1/datasets/download/hmavrodiev/london-bike-sharing-dataset

# Ensure the target directory exists
mkdir -p ./raw

# Unpack all files into /raw, overwriting if necessary
unzip -o ./london-bike-sharing-dataset.zip -d ./raw

echo "Dataset downloaded and extracted to /raw successfully."

# Remove the downloaded zip file
rm ./london-bike-sharing-dataset.zip
