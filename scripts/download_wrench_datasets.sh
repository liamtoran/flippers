#!/bin/bash

# Many thanks to the awesome wrench project
# See https://github.com/JieyuZ2/wrench#-available-datasets 
echo "Downloading wrench datasets from gdrive"
echo "All credits for creating the wrench benchmark goes to https://github.com/JieyuZ2/wrench#-available-datasets"

# Download the file
FILE_ID="19wMFmpoo_0ORhBzB6n16B1nRRX508AnJ"
FILE_NAME="data.zip"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILE_ID}" -o ${FILE_NAME}

# Unzip the file
unzip ${FILE_NAME}
rm ${FILE_NAME}
rm ./cookie
echo "Downloading of https://github.com/JieyuZ2/wrench#-available-datasets is complete!"
