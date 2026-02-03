#!/bin/bash

DATASET_ROOT="dataset"
# ==========================================

echo "Generating mazes into ${DATASET_ROOT}..."

python data_utils/maze_generator.py --grid-n 3 --n-mazes 1250 --min-path-length 2 --max-path-length 6 --output-dir "./${DATASET_ROOT}/maze3x3" --file-prefix "maze3" -q
python data_utils/maze_generator.py --grid-n 4 --n-mazes 1250 --min-path-length 2 --max-path-length 8 --output-dir "./${DATASET_ROOT}/maze4x4" --file-prefix "maze4" -q
python data_utils/maze_generator.py --grid-n 5 --n-mazes 1250 --min-path-length 2 --max-path-length 10 --output-dir "./${DATASET_ROOT}/maze5x5" --file-prefix "maze5" -q 
python data_utils/maze_generator.py --grid-n 6 --n-mazes 1250 --min-path-length 2 --max-path-length 12 --output-dir "./${DATASET_ROOT}/maze6x6" --file-prefix "maze6" -q
python data_utils/maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 2 --max-path-length 14 --output-dir "./${DATASET_ROOT}/maze7x7" --file-prefix "maze7" -q
python data_utils/maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 2 --max-path-length 16 --output-dir "./${DATASET_ROOT}/maze8x8" --file-prefix "maze8" -q
python data_utils/maze_generator.py --grid-n 6 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze6x6_ood" --file-prefix "maze6ood" -q
python data_utils/maze_generator.py --grid-n 5 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze5x5_ood" --file-prefix "maze5ood" -q
python data_utils/maze_generator.py --grid-n 7 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze7x7_ood" --file-prefix "maze7ood" -q
python data_utils/maze_generator.py --grid-n 8 --n-mazes 250 --min-path-length 13 --max-path-length 18 --output-dir "./${DATASET_ROOT}/maze8x8_ood" --file-prefix "maze8ood" -q

echo "Organizing training data..."

for ((i=3; i<=6; i++)); do
    source_dir="${DATASET_ROOT}/maze${i}x${i}"
    target_dir="${DATASET_ROOT}/maze_train"

    mkdir -p "$target_dir"

    for ((x=1; x<=1000; x++)); do
        num=$(printf "%04d" $x)

        mp4_file="maze${i}_${num}.mp4"

        if [ -f "$source_dir/$mp4_file" ]; then
            mv "$source_dir/$mp4_file" "$target_dir/"
        fi
        
        png_file="maze${i}_${num}_00.png"
        if [ -f "$source_dir/$png_file" ]; then
            mv "$source_dir/$png_file" "$target_dir/"
        fi

        png_file_icon="maze${i}_${num}_icon.png"
        if [ -f "$source_dir/$png_file_icon" ]; then
            mv "$source_dir/$png_file_icon" "$target_dir/"
        fi
    done
done

python data_utils/metadata_gen.py --dir "${DATASET_ROOT}/maze_train"

echo "Train dataset created at ${DATASET_ROOT}/maze_train"
echo "Organizing test data..."

mkdir -p "${DATASET_ROOT}/maze_test"
mv "${DATASET_ROOT}"/{maze3x3,maze4x4,maze5x5,maze6x6,maze7x7,maze8x8,maze5x5_ood,maze6x6_ood,maze7x7_ood,maze8x8_ood} "${DATASET_ROOT}/maze_test"

echo "Test dataset created at ${DATASET_ROOT}/maze_test"
echo "All Done."