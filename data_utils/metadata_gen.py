import sys
import os
import argparse

parser = argparse.ArgumentParser(description="Generate metadata.csv for maze dataset")
parser.add_argument("--dir", type=str, required=True, help="Path to the maze_train directory")
args = parser.parse_args()

train_dir = args.dir

if not os.path.exists(train_dir):
    print(f"Error: Directory {train_dir} does not exist.")
    sys.exit(1)

prompt = "Create a 2D animation based on the provided image of a maze.\
    The agent slides smoothly along the white path, stopping perfectly on the red flag and then acquiring a trophy.\
    The agent never slides or crosses into the black segments of the maze.\
    The camera is a static, top-down view showing the entire maze.\
    Maze:\
    * The maze paths are white, the walls are black.\
    * The agent starts from origin, represented by a green circle.\
    * The agent slides smoothly along the white path.\
    * The agent never slides or crosses into the black segments of the maze.\
    * The agent stops perfectly on the red flag, acquiring a trophy thereafter.\
    Scene:\
    * No change in scene composition.\
    * No change in the layout of the maze.\
    * The agent travels along the path without speeding up or slowing down.\
    Camera:\
    * Static camera.\
    * No zoom.\
    * No pan.\
    * No glitches, noise, or artifacts."

metadata_path = os.path.join(train_dir, "metadata.csv")

print(f"Generating metadata to: {metadata_path}")

with open(metadata_path, "w") as f:
    f.write("video,prompt,input_image,icon_image\n")

for grid_n in [3, 4, 5, 6]:
    for i in range(1, 1001):
        filename = f"maze{grid_n}_{i:04d}.mp4"
        
        framename = f"maze{grid_n}_{i:04d}_00.png"
        iconname = f"maze{grid_n}_{i:04d}_icon.png"

        with open(metadata_path, "a") as f:
            f.write(f"{filename},\"{prompt}\",{framename},{iconname}\n")

print("Metadata generation complete.")