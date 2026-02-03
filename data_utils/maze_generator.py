# Generate mp4 and first frame png for mazes
# Parameters: --output-dir, --maze-size, --n-mazes, --min-path-length, --max-path-length, --create-algo, --quiet
# Output files are saved in the specified output directory

from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators, get_maze_with_solution

import sys
import os
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from plot_maze import MazePlot
import numpy as np
import imageio
import cv2
import random
import csv  
from concurrent.futures import ProcessPoolExecutor

sys.setrecursionlimit(20000)

default_output_dir = "dataset/maze3x3"
default_maze_size = 3
default_n_mazes = 10
default_maze_ctor = 'gen_kruskal'

parser = argparse.ArgumentParser(description="Generate mazes and export to mp4 and png")
parser.add_argument("--output-dir", "-o", type=str, default=default_output_dir,
                    help="Directory to save output files")
parser.add_argument("--grid-n", "-s", type=int, default=default_maze_size,
                    help="Size of the maze (grid_n by grid_n)")
parser.add_argument("--n-mazes", "-n", type=int, default=default_n_mazes,
                    help="Number of mazes to generate, default=10")
parser.add_argument("--min-path-length", "-l", type=int, help="(Optional) Minimum path length")
parser.add_argument("--max-path-length", "-r", type=int, help="(Optional) Maximum path length")
parser.add_argument("--create-algo", "-c", type=str, default=default_maze_ctor,
                    help="(Optional) Maze generation algorithm, default='gen_kruskal'")
parser.add_argument("--quiet", "-q", help="Quiet mode, no output", action="store_true")

# =========================================================================
# Parameters for Icons
# =========================================================================
parser.add_argument("--icon_origin", type=str, help="Origin icon file path", default=None)
parser.add_argument("--icon_target", type=str, help="Target icon file path", default="dataset_icons/red_circle.png")
parser.add_argument("--icon_agent", type=str, help="Agent icon file path directory", default="dataset_icons/train")
parser.add_argument("--icon_end", type=str, help="End icon file path", default=None)
parser.add_argument("--file-prefix", "-p", type=str, default="maze", help="(Optional) File prefix for output files")

args = parser.parse_args()

# -------------------------------------------------------------------------
# 1. Prepare Agent Candidates
# -------------------------------------------------------------------------
if os.path.isdir(args.icon_agent):
    agent_candidates = [
        os.path.join(args.icon_agent, f) 
        for f in os.listdir(args.icon_agent) 
        if f.lower().endswith('.png')
    ]
    if not agent_candidates:
        raise ValueError(f"No png files found in directory: {args.icon_agent}")
    agent_candidates.sort()
    print(f"Found {len(agent_candidates)} agent icons in {args.icon_agent}")
else:
    agent_candidates = [args.icon_agent]

# =========================================================================
# Construct Icon Dictionary
# =========================================================================
icon_path_set = {
    "origin": None,
    "target": args.icon_target,
    "end": None
}

# -------------------------------------------------------------------------
# 2. Generate Maze Data Structure
# -------------------------------------------------------------------------
output_dir = args.output_dir
grid_n = args.grid_n
n_mazes = args.n_mazes
min_path_length = args.min_path_length if args.min_path_length else 2
max_path_length = args.max_path_length if args.max_path_length else 2 * grid_n

file_prefix = args.file_prefix if args.file_prefix else f"maze{grid_n}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

if max_path_length > grid_n * grid_n:
    print(f"max_path_length must be <= grid_n*grid_n={grid_n * grid_n}")
    exit(0)
if min_path_length < 2 or min_path_length >= max_path_length:
    print(f"min_path_length must be >=2 and < max_path_length={max_path_length}")
    exit(0)

maze_ctor = args.create_algo
target_lengths = [(i % (max_path_length - min_path_length + 1)) + min_path_length for i in range(n_mazes)]
need_lengths = {length: target_lengths.count(length) for length in set(target_lengths)}
mazes_with_length = [list() for _ in range(max_path_length + 1)]
mazes_got = 0

random.seed(0)
np.random.seed(0)

# Generation Loop
while mazes_got < n_mazes:
    newmaze = get_maze_with_solution(maze_ctor, (grid_n, grid_n))
    now_length = len(newmaze.solution)

    if now_length > max_path_length:
        continue
    if now_length not in need_lengths:
        continue

    if len(mazes_with_length[now_length]) < need_lengths[now_length]:
        mazes_with_length[now_length].append(newmaze)
        mazes_got += 1

dataset = []
count_lengths = [0 for i in range(max_path_length + 1)]
for i in range(n_mazes):
    target = target_lengths[i]
    dataset.append(mazes_with_length[target][count_lengths[target]])
    count_lengths[target] += 1


# -------------------------------------------------------------------------
# 3. Define Export Functions
# -------------------------------------------------------------------------

def export_mp4(m, icon_path_set, output_file, fps=15):
    """
    Export logic:
    1. _00.png: Input First Frame (Agent visible)
    2. .mp4: Full Video (Agent visible)
    """
    mp = MazePlot(m, icon_path_set=icon_path_set)
    
    # === Step 1: Input First Frame ===
    frames_clean = mp.plot_continuous(frames_num=1, hide_agent=False)
    
    frames_clean[0].canvas.draw()
    img_clean = np.array(frames_clean[0].canvas.renderer.buffer_rgba())[:, :, :3]
    img_clean_bgr = cv2.cvtColor(img_clean, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{output_dir}/{output_file}_00.png', img_clean_bgr)
    
    plt.close(frames_clean[0])

    # === Step 2: GT Video ===
    frames_video = mp.plot_continuous(frames_num=81, hide_agent=False)
    
    def create_video(frames, output_file='output.mp4', fps=15):
        images = []
        for fig in frames:
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
            images.append(img)
            plt.close(fig)
        imageio.mimsave(output_file, images, fps=fps)

    create_video(frames_video, f'{output_dir}/{output_file}.mp4', fps=fps)
    
def process_single_maze(i, maze):
    # Reset random seed for multiprocessing
    random.seed(i)
    np.random.seed(i)

    # 1. Randomly select Agent
    current_agent_path = random.choice(agent_candidates)
    
    # 2. Build Icon Set
    current_icon_path_set = icon_path_set.copy()
    current_icon_path_set["agent"] = current_agent_path
    
    # 3. Prepare filename
    current_file_prefix = f"maze{grid_n}"
    video_filename = f'{current_file_prefix}_{i + 1:04d}'
    
    # 4. Export video and first frame
    export_mp4(maze, current_icon_path_set, video_filename, fps=15)

    # 5. RETURN Metadata (Do not write to file here)
    # Convert numpy array to list for JSON serialization compatibility
    solution_py = np.array(maze.solution).tolist()
    
    return {
        "filename": video_filename,
        "grid_size": int(grid_n),
        "agent_icon": current_agent_path,
        "solution": str(solution_py), 
        "start": str(solution_py[0]),
        "end": str(solution_py[-1])
    }

# Wrapper for multiprocessing
def task_wrapper(args):
    return process_single_maze(*args)


# -------------------------------------------------------------------------
# 4. Main Program
# -------------------------------------------------------------------------
if __name__ == '__main__':
    tasks = [(i, maze) for i, maze in enumerate(dataset)]

    from tqdm import tqdm
    print(f"Making {n_mazes} {grid_n} by {grid_n} mazes to {output_dir}/ using Multiprocessing...")

    results = []

    with ProcessPoolExecutor(max_workers=16) as executor:
        for res in tqdm(executor.map(task_wrapper, tasks), total=len(tasks), desc="Generating"):
            results.append(res)

    csv_file = os.path.join(output_dir, "metadata.csv")
    print(f"Writing metadata to {csv_file}...")
    
    if results:
        fieldnames = ["filename", "grid_size", "agent_icon", "solution", "start", "end"]
        
        with open(csv_file, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    print("Done.")