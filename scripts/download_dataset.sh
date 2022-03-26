#!/bin/bash
# Description: Download "UCI HAR Dataset" and extract it to data_dir_path.

print_usage() {
  echo "Usage: $0 data_dir_path"
  echo "Example: $0 /path/to/har-lastm/data"
}

if [[ $# -ne 1 && !("$1" == "--help" || "$1" == "-h") ]];
then
  print_usage
  exit 1
elif [[ "$1" == "--help" || "$1" == "-h" ]];
then
  print_usage
  exit 0
fi 

DATA_DIR_PATH=$1

DATASET_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
DATASET_FILENAME="UCI_HAR_Dataset.zip"

echo ">> Create a data folder in ${DATA_DIR_PATH}"

curl -o "${DATASET_FILENAME}" "${DATASET_URL}"

echo ">> Extract ${DATASET_FILENAME} to ${DATA_DIR_PATH}"
mkdir -p "${DATA_DIR_PATH}"
unzip "${DATASET_FILENAME}" -d "${DATA_DIR_PATH}"

# Rename a folder for simplicity
mv "${DATA_DIR_PATH}/UCI HAR Dataset" "${DATA_DIR_PATH}/har_dataset"


# Delete unncessary files and folders
echo ">> Cleaning unncessary files and folders"
rm -r "${DATA_DIR_PATH}/__MACOSX"
rm "${DATASET_FILENAME}"