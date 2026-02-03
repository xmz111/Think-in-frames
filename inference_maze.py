import sys
import argparse
import os
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(root_dir)
#from prompts import MAZE_PROMPT as prompt
from safetensors.torch import load_file
import torch
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from pathlib import Path

prompt = "Create a 2D animation based on the provided image of a maze.     The custom character slides smoothly along the white path, stopping perfectly on the red circle destination.     The character never slides or crosses into the black segments of the maze.     The camera is a static, top-down view showing the entire maze.     Maze:     * The maze paths are white, the walls are black.     * The character starts from its initial position.     * The character slides smoothly along the white path.     * The character never slides or crosses into the black segments of the maze.     * The character stops perfectly on the red circle.     Scene:     * No change in scene composition.     * No change in the layout of the maze.     * The character travels along the path without speeding up or slowing down.     Camera:     * Static camera.     * No zoom.     * No pan.     * No glitches, noise, or artifacts."

def generate_prompt_from_filename(filename):
    try:
        base = os.path.splitext(filename)[0]

        if "_" in base:
            if base.endswith("_00"): 
                 base = base.rsplit('_', 2)[0]
            else:
                 base = base.rsplit('_', 1)[0]
        parts = base.split('-')
        
        # 3. 提取实体 (index 0 是 'maze5'，跳过)
        origin = parts[1].replace('_', ' ')
        target = parts[2].replace('_', ' ')
        agent  = parts[3].replace('_', ' ')
        end    = parts[4].replace('_', ' ')

        agent = 'blue star'
        # 4. 填充模板
        return PROMPT_TEMPLATE.format(
            AGENT=agent, TARGET=target, ORIGIN=origin, END=end
        )
    except Exception as e:
        print(f"Warning: Filename parse error '{filename}': {e}. Using default.")
        return "Create a 2D animation based on the provided image of a maze."

def get_pipe():
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B",
                        origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
            ModelConfig(model_id="Wan-AI/Wan2.2-TI2V-5B", origin_file_pattern="Wan2.2_VAE.pth", offload_device="cpu"),
        ],
    )

    pipe.load_lora(pipe.dit, f"epoch-19.safetensors", alpha=1)
    pipe.enable_vram_management()
    return pipe


def run(args, pipe, file_path, quiet, num_frames, prompt):
    input_image = Image.open(file_path)
    video = pipe(
        prompt=prompt,
        num_frames=num_frames,
        negative_prompt="",
        input_image=input_image,
        seed=0, tiled=True,
    )
    print(file_path)
    if args.output_dir:
        output_path = args.output_dir + file_path.split('/')[-1][:-4] + "_" + str(num_frames) + "inference.mp4"
    else:
        output_path = file_path[:-4] + "_" + str(num_frames) + "inference.mp4"
    print(output_path)
    save_video(video, output_path, fps=15, quality=5)
    if not quiet:
        tqdm.write(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Inference')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('file', nargs='?', help='Input data file (PNG)')
    group.add_argument('--input-dir', metavar='DIR', dest='dir', help='Process all PNG files in the directory')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output messages')
    parser.add_argument('--num_frames', type=int, default=81, help='num_frames')
    parser.add_argument('--output-dir', type=str, default=None, help='output dir of videos')
    args = parser.parse_args()

    pipe = get_pipe()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.file:
        if not args.file.endswith('.png'):
            print(f"Error: {args.file} is not a PNG file.")
            sys.exit(1)
        if not os.path.isfile(args.file):
            print(f"Error: {args.file} does not exist.")
            sys.exit(1)
        run(args, pipe, args.file, args.quiet, args.num_frames, prompt)

    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"Error: {args.dir} does not exist.")
            sys.exit(1)

        data = []
        for item in os.listdir(args.dir):
            if item.endswith('_00.png'):
                file_path = os.path.join(args.dir, item)
                if os.path.isfile(file_path):
                    data.append(file_path)

        data.sort()

        for item in tqdm(data, desc=f"Processing {args.dir}", unit="file", disable=args.quiet):
            run(args, pipe, item, args.quiet, args.num_frames, prompt)


if __name__ == '__main__':
    main()
