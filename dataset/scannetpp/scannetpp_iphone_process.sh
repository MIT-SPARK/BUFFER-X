#!/bin/bash
set -e  # If any command fails, exit immediately

echo "1. Running prepare_iphone_data.py..."
python iphone/prepare_iphone_data.py iphone/prepare_iphone_data.yml

echo "2. Running copy_iphone_dir.py..."
python iphone/copy_iphone_dir.py

echo "3. Compiling iPhone data with compile.sh..."
bash iphone/compile.sh

echo "4. Running ./iphone/scannetpp (make sure it is an executable)..."
./iphone/scannetpp

echo "5. Generating pairs with pair_gen_iphone.py..."
python iphone/pair_gen_iphone.py

echo "✅ All iPhone data preparation steps completed successfully."
