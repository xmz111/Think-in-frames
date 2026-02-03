import cv2
import numpy as np
from scipy.interpolate import interp1d
import os

PROCESS_W, PROCESS_H = 320, 320

def get_clean_background(cap, num_frames=30):

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        return None

    sample_indices = np.linspace(0, total_frames - 1, min(total_frames, num_frames), dtype=int)
    frames = []
    
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (PROCESS_W, PROCESS_H))
            frames.append(frame)
            
    if not frames:
        return None

    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    return median_frame

def process_video_motion(video_path, debug_output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    bg_frame = get_clean_background(cap)
    if bg_frame is None:
        return []

    writer = None
    if debug_output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(debug_output_path, fourcc, 15, (PROCESS_W, PROCESS_H))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    trajectory = []
    last_center = None

    SEARCH_RADIUS = 60 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (PROCESS_W, PROCESS_H))

        diff = cv2.absdiff(bg_frame, frame_resized)

        diff_gray = np.max(diff, axis=2) 

        thresh = cv2.threshold(diff_gray, 30, 255, cv2.THRESH_BINARY)[1]
        thresh = thresh.astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=3) 

        if last_center is not None:
            mask = np.zeros_like(thresh)
            cv2.circle(mask, last_center, SEARCH_RADIUS, 255, -1)
            thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_center = None
        
        if cnts:
            valid_cnts = [c for c in cnts if cv2.contourArea(c) > 20]
            
            if valid_cnts:
                if last_center is None:
                    c = max(valid_cnts, key=cv2.contourArea)
                    best_center = get_contour_center(c)
                else:
                    candidates = [get_contour_center(c) for c in valid_cnts]
                    dists = [np.linalg.norm(np.array(p) - np.array(last_center)) for p in candidates]
                    min_idx = np.argmin(dists)
                    
                    if dists[min_idx] < 80:
                        best_center = candidates[min_idx]
                    else:
                        best_center = last_center

        if best_center:
            trajectory.append(best_center)
            last_center = best_center
        else:
            if last_center:
                trajectory.append(last_center)
            else:
                pass 

        if writer:
            vis = frame_resized.copy()
            if last_center:
                cv2.circle(vis, last_center, SEARCH_RADIUS, (0, 255, 0), 1)
            if best_center:
                cv2.circle(vis, best_center, 4, (0, 0, 255), -1)
            elif trajectory: 
                cv2.circle(vis, trajectory[-1], 4, (0, 255, 255), -1)

            mask_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            vis[0:80, 0:80] = cv2.resize(mask_vis, (80, 80))
            writer.write(vis)

    cap.release()
    if writer:
        writer.release()

    if not trajectory:
        return []

    return smooth_trajectory(trajectory)

def get_contour_center(c):
    M = cv2.moments(c)
    if M["m00"] != 0:
        return (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
    else:
        x, y, w, h = cv2.boundingRect(c)
        return (x + w//2, y + h//2)

def smooth_trajectory(traj, window=5):
    if len(traj) < window: return traj
    arr = np.array(traj)
    kernel = np.ones(window) / window

    smooth_x = np.convolve(arr[:, 0], kernel, mode='valid') 
    smooth_y = np.convolve(arr[:, 1], kernel, mode='valid')

    pad_head = arr[:window//2]
    pad_tail = arr[-window//2 + 1:] if window % 2 == 0 else arr[-(window//2):]

    new_x = np.concatenate([pad_head[:,0], smooth_x, pad_tail[:,0]])
    new_y = np.concatenate([pad_head[:,1], smooth_y, pad_tail[:,1]])

    if len(new_x) != len(traj):
        smooth_x = np.convolve(arr[:, 0], kernel, mode='same')
        smooth_y = np.convolve(arr[:, 1], kernel, mode='same')
        smooth_x[:window//2] = arr[:window//2, 0]
        smooth_y[:window//2] = arr[:window//2, 1]
        return list(zip(smooth_x, smooth_y))
        
    return list(zip(new_x, new_y))

def get_traj(gt_path, stu_path, icon_path=None):
    debug = False
    gt_d = gt_path.replace(".mp4", "_debug.mp4") if debug else None
    stu_d = stu_path.replace(".mp4", "_debug.mp4") if debug else None
    return process_video_motion(gt_path, gt_d), process_video_motion(stu_path, stu_d)

def interpolate_trajectory(traj, m):
    traj = np.array(traj)
    if len(traj) == 0: return np.zeros((m, 2))
    unique_indices = []
    for i in range(len(traj)):
        if i == 0 or not np.array_equal(traj[i], traj[i - 1]):
            unique_indices.append(i)
    traj = traj[unique_indices]
    if len(traj) < 2:
        if len(traj) == 1: return np.full((m, 2), traj[0])
        return np.zeros((m, 2))
    distances = np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))
    cumulative_dist = np.concatenate(([0], np.cumsum(distances)))
    if cumulative_dist[-1] == 0: return np.full((m, 2), traj[0])
    interp_func = interp1d(cumulative_dist, traj, axis=0, kind='linear')
    new_distances = np.linspace(0, cumulative_dist[-1], m)
    return interp_func(new_distances)

def compare_traj(traj1, traj2, traj_len=100):
    t1 = interpolate_trajectory(traj1, traj_len)
    t2 = interpolate_trajectory(traj2, traj_len)
    dists = np.linalg.norm(t1 - t2, axis=1)
    return np.argmax(dists), np.max(dists), dists, t1, t2