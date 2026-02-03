import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import argparse
from tqdm import tqdm
from utils_icon import get_traj, compare_traj


def plot_density(data, filename='density_plot.png', save_path='./', div=[10, 20, 30], title='Density Plot'):
    plt.figure(figsize=(10, 6))
    boundaries = div
    sns.histplot(data, kde=True, stat="density", alpha=0.7)
    ranges = [(-np.inf, boundaries[0]), (boundaries[0], boundaries[1]), (boundaries[1], boundaries[2]), (boundaries[2], np.inf)]
    plt.xlabel('maxdistance')
    plt.ylabel('density')
    plt.title(title)
    plt.grid(alpha=0.3)
    for boundary in boundaries:
        plt.axvline(x=boundary, color='black', linestyle='--', alpha=0.7, linewidth=1)
    data = np.array(data)
    data = np.sort(data)
    proportions_text = f"Proportions: (total={len(data)})\n"
    for i, (start, end) in enumerate(ranges):
        if start == -np.inf:
            count = np.sum(data <= end)
            range_desc = f"≤{end}"
        elif end == np.inf:
            count = np.sum(data > start)
            range_desc = f">{start}"
        else:
            count = np.sum((data > start) & (data <= end))
            range_desc = f"{start}-{end}"
        proportion = count / len(data)
        proportions_text += f"{range_desc}: {proportion:.1%}\n"
    plt.text(0.95, 0.95, proportions_text, transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=10, fontfamily='monospace')
    os.makedirs(save_path, exist_ok=True)
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def eval_dis(dis, div):
    if dis > div[2]:
        evaluation = "Significantly Different"
    elif dis > div[1]:
        evaluation = "Slightly Similar"
    elif dis > div[0]:
        evaluation = "Moderately Similar"
    else:
        evaluation = "Highly Similar"
    return evaluation

def eval(
    args,
    dir,
    output_dir,
    epoch_name,
    div : list,
    threshold,
    inter_traj_len=5000,
    omit=False,
    num_frames=81,
):
    
    items = os.listdir(dir)
    similarity_groups = {'Highly Similar': [], 'Moderately Similar': [], 'Slightly Similar': [], 'Significantly Different': []}
    max_distances = []
    PRs = []
    png_items = []

    if args.target_dir:
        target_dir = args.target_dir
    else:
        target_dir = dir
    
    for item in items:
        item_path = os.path.join(dir, item)
        if os.path.isfile(item_path) and item.lower().endswith('.png'):
            parts = item.split('_00.png')
            name = parts[0]
            gt_file_path = os.path.join(dir, f"{name}.mp4")

            if epoch_name != "":
                if num_frames != 81:
                    inference_file_path = os.path.join(target_dir, f"{name}_{epoch_name}_{num_frames}inference.mp4")
                else:
                    p1 = os.path.join(target_dir, f"{name}_{epoch_name}_inference.mp4")
                    p2 = os.path.join(target_dir, f"{name}_{epoch_name}_{num_frames}inference.mp4")
                    inference_file_path = p2 if os.path.exists(p2) else p1
            else:
                inference_file_path = os.path.join(target_dir, f"{name}inference.mp4")
            
            if os.path.isfile(inference_file_path) and os.path.isfile(gt_file_path):
                png_items.append(item)

    total_num = len(png_items)
    if total_num == 0:
        return

    def colored_similarity_output(filename, similarity_type):
        colors = {'Moderately Similar': '\033[92m', 'Slightly Similar': '\033[93m', 'Significantly Different': '\033[91m'}
        color_code = colors.get(similarity_type, '')
        reset_code = '\033[0m'
        if similarity_type in colors:
            return f"{color_code}{filename}: {similarity_type}{reset_code}"
        return None

    for item in tqdm(png_items, desc=f"Processing {dir}", total=total_num,
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % ('\033[94m', '\033[0m')):
        parts = item.split('_00.png')
        name = parts[0]
        file1 = f"{name}.mp4"

        if epoch_name:
            candidates = [
                f"{name}_{epoch_name}_{num_frames}inference.mp4",
                f"{name}_{epoch_name}_inference.mp4"
            ]
            file2 = candidates[0]
            for c in candidates:
                if os.path.exists(os.path.join(target_dir, c)):
                    file2 = c
                    break
        else:
            file2 = f"{name}_inference.mp4"
        
        raw_expert, raw_student = get_traj(f"{dir}/{file1}", f"{target_dir}/{file2}")

        if not raw_expert or not raw_student:
            tqdm.write(f"Warning: Could not extract trajectory from {file1} or {file2}")
            continue

        if len(raw_expert) != len(raw_student):
            student_indices = np.linspace(0, len(raw_student) - 1, len(raw_student))
            target_indices = np.linspace(0, len(raw_student) - 1, len(raw_expert))
            student_x = np.interp(target_indices, student_indices, [p[0] for p in raw_student])
            student_y = np.interp(target_indices, student_indices, [p[1] for p in raw_student])
            student = list(zip(student_x, student_y))
            expert = raw_expert
        else:
            expert, student = raw_expert, raw_student

        max_index, max_distance, dis, _, __ = compare_traj(expert, student, inter_traj_len)

        first_diff = inter_traj_len
        for i in range(len(dis)):
            if dis[i] > threshold:
                first_diff = i
                break
        
        pr_denominator = len(dis) if len(dis) > 0 else 1
        metrics = {
            "max_distance": max_distance,
            "first_diff": first_diff,
            "PR": first_diff / pr_denominator,
            "similarity_evaluation": eval_dis(max_distance, div)
        }
        PRs.append(metrics['PR'])
        max_distances.append(max_distance)
        similarity_groups[metrics['similarity_evaluation']].append((file2, metrics))

        colored_output = colored_similarity_output(file2, metrics['similarity_evaluation'])
        if colored_output:
            tqdm.write(colored_output)

    plot_density(max_distances, "distances_plot", output_dir, div, title=f"{dir} {epoch_name}")
    for evaluation, files in similarity_groups.items():
        filename = f"{evaluation.replace(' ', '_').lower()}_files.txt"
        filename = f"{output_dir}/{filename}"
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"{evaluation} Files\n")
            f.write("=" * 50 + "\n")
            files_sorted=sorted(files, key=lambda x: x[0])
            f.write(f"Total: {len(files)} files.\n\n")
            for file in files_sorted:
                f.write(f"{file[0]}: dist={file[1]['max_distance']:.2f}, PR={file[1]['PR']:.2f}\n")
        if not omit:
            print(f"Generated file: {filename}")
    with open(f"{output_dir}/summary.txt", 'w', encoding='utf-8') as f:
        pr = sum(PRs) / len(PRs) if PRs else 0
        em = PRs.count(1) / len(PRs) if PRs else 0
        f.write(f"PR: {pr:.4f}\n")
        f.write(f"EM: {em:.4f}\n")
    if not omit:
        print(f"Generated file: {output_dir}/summary.txt")


if __name__ == "__main__":
    default_dir = "./"
    default_output_dir = "./result"
    default_epoch_name = "00"
    default_div = [10, 20, 30]
    default_threshold = 20
    parser = argparse.ArgumentParser(description='Evaluate video reasoning results.')
    parser.add_argument('--input-dir', '-i', type=str, default=default_dir, help='Evaluation data input directory.')
    parser.add_argument('--target-dir', type=str, default=None, help='Directory of inference videos.')
    parser.add_argument('--output-dir', '-o', type=str, default=default_output_dir, help='Evaluation result output directory')
    parser.add_argument('--epoch-name', '-e', type=str, default=default_epoch_name, help='Epoch name in filename.')
    parser.add_argument('--div', '-d', type=str, default=','.join(map(str, default_div)), help='Division points')
    parser.add_argument('--quiet', '-q', action='store_true', help='Quiet mode')
    parser.add_argument('--threshold', '-t', type=int, default=default_threshold, help='Pixel distance threshold.')
    parser.add_argument('--num_frames', type=int, default=81, help='num_frames')
    args = parser.parse_args()
    
    dir_path = args.input_dir
    output_dir = args.output_dir
    epoch_name = args.epoch_name if args.epoch_name else ""
    div = list(map(int, args.div.split(','))) if args.div else default_div
    threshold = args.threshold
    omit = args.quiet

    if not omit:
        print(f"Evaluation data input directory: {dir_path}")
        print(f"Evaluation result output directory: {output_dir}")
    
    eval(args, dir_path, output_dir, epoch_name, div, omit=omit, threshold=threshold, num_frames=args.num_frames)