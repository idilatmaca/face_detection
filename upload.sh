#!/bin/bash
# ==============================================================================
# Local Azure Folder Upload
# ==============================================================================

# --- CONFIGURATION ---
# IMPORTANT: Update this path to the actual folder on your local computer
SOURCE_DIR="/media/syn3090/16c65c3c-0381-482a-a5df-5f99340fac70/deneme/light_resnet/All" 

STORAGE_ACCOUNT="syntonymdatastorage"
CONTAINER_NAME="datasets"
# This is where it will land inside the container
BLOB_DESTINATION_PATH="coone-face-detection"

echo "========================================"
echo "Azure Local Upload"
echo "========================================"
echo "Source:      $SOURCE_DIR"
echo "Destination: $CONTAINER_NAME/$BLOB_DESTINATION_PATH"
echo ""

# 1. Check if Azure CLI is installed
if ! command -v az &> /dev/null; then
    echo "❌ ERROR: 'az' command not found."
    echo "Please install the Azure CLI first (brew install azure-cli OR apt-get install azure-cli)."
    exit 1
fi

# 2. Check if Source Directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ ERROR: Source directory not found!"
    echo "   Path looked for: $SOURCE_DIR"
    echo "   Please edit the 'SOURCE_DIR' variable in this script."
    exit 1
fi

# 3. Check if logged in (simple check)
echo "Checking Azure login status..."
az account show &> /dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  You are not logged in."
    echo "   Please run: 'az login' and follow the browser prompts, then run this script again."
    exit 1
fi

# 4. File Counting (Optional, just for info)
echo "Counting files..."
FILE_COUNT=$(find "$SOURCE_DIR" -type f | wc -l)
echo "Total files to upload: $FILE_COUNT"
echo ""

echo "Starting upload..."
echo "========================================"

# 5. The Upload Command
# We removed the custom progress monitor because running this locally 
# allows 'az' to show its own native progress bar.
az storage blob upload-batch \
    --account-name "${STORAGE_ACCOUNT}" \
    --source "${SOURCE_DIR}" \
    --destination "${CONTAINER_NAME}" \
    --destination-path "${BLOB_DESTINATION_PATH}" \
    --auth-mode login \
    --overwrite true \
    --max-connections 32

EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ SUCCESS: Upload complete."
else
    echo "❌ FAILED: Something went wrong. Exit code $EXIT_CODE"
fi
echo "Finished: $(date)"
echo "========================================"