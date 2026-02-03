# Think-in-frames
## Environment
```
git clone https://github.com/xmz111/Think-in-frames
cd Think-in-frames
conda create -n think_in_frames python=3.12 -c conda-forge -y
conda activate think_in_frames
pip install -r requirements.txt
pip install -e maze-dataset
```
## Data Generation
```
bash data_utils/data_gen_train.sh
bash data_utils/data_gen_ood_test.sh
```
## Inference
```
python inference.py --input-dir=dataset/maze_test/maze5x5_ood/ --output-dir=test_videos/ --num_frames=81
```
## Evaluation
```
python eval.py --input-dir=dataset/maze_test/maze5x5_ood/ --target-dir=test_videos/ --output-dir=test_videos/ --num_frames=81
```
## Acknowledgements

- [Wan](https://github.com/Wan-Video/Wan2.2): Powerful open-source video diffusion models used as base models.
- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio): Video diffusion model training.
- [maze-dataset](https://github.com/understanding-search/maze-dataset): Data generation for maze reasoning tasks.
- [MiniVeo3-Reasoner](https://github.com/thuml/MiniVeo3-Reasoner): Provide good codebase.
