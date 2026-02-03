#!/bin/bash

DATASET_ROOT="dataset_ood_test"
# ==========================================

echo "Generating mazes into ${DATASET_ROOT}..."

python data_utils/maze_generator.py --grid-n 5 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze5x5_ood" --file-prefix "maze5ood" -q --icon_agent dataset_icons/test
python data_utils/maze_generator.py --grid-n 6 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze6x6_ood" --file-prefix "maze6ood" -q --icon_agent dataset_icons/test
python data_utils/maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze7x7_ood" --file-prefix "maze7ood" -q --icon_agent dataset_icons/test
python data_utils/maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze8x8_ood" --file-prefix "maze8ood" -q --icon_agent dataset_icons/test

echo "All Done."