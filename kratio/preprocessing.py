"""
preprocessing.py
----------------
Data loading and preprocessing for EyeLink Waldo task recordings.
"""

import os
import numpy as np
import pandas as pd


# Data Loading
def load_waldo_directory(directory, max_files=15):
    all_data = []
    file_count = 0

    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".txt"):
            file_count += 1
            if file_count > max_files:
                break
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            numeric_data = [line.strip().split() for line in lines if not line.startswith("MSG")]
            if not numeric_data:
                continue
            df = pd.DataFrame(numeric_data)
            df.columns = [f"col_{i}" for i in range(len(df.iloc[0]))]
            df = df.apply(pd.to_numeric, errors='coerce')
            all_data.append(df)

    print(f"Total files loaded: {len(all_data)}")
    for i, df in enumerate(all_data):
        print(f"  File {i+1}: Shape = {df.shape}")
    return all_data


# NaN cleaning (250-sample margin)


def _nan_mask_with_margin(xl, yl, xr, yr, margin=250):
    Nan_array = (
        np.logical_not(np.isnan(xl)) &
        np.logical_not(np.isnan(yl)) &
        np.logical_not(np.isnan(xr)) &
        np.logical_not(np.isnan(yr))
    )
    new_nan_array = Nan_array.copy()
    for j in range(1, margin):
        if j < len(new_nan_array):
            new_nan_array[:-j] *= Nan_array[j:]
    Nan_array = new_nan_array
    for j in range(1, margin):
        if j < len(new_nan_array):
            new_nan_array[j:] *= Nan_array[:-j]
    return new_nan_array


# Preprocessing pipeline


def preprocess_waldo(all_data, remove_nanbefore=250):
    time_list = []
    x_left_list, y_left_list = [], []
    x_right_list, y_right_list = [], []
    cat_left_list, cat_right_list = [], []
    saccade_list, fixation_list = [], []

    for df in all_data:
        arr = np.swapaxes(np.array(df), 0, 1)

        t   = arr[0]
        xl  = arr[1];  yl  = arr[2]
        xr  = arr[4];  yr  = arr[5]
        cl  = arr[8];  cr  = arr[9]

        fix  = np.where(cl == 1, 1, 0)
        sac  = np.where(cl == 2, 1, 0)

        mask = _nan_mask_with_margin(xl, yl, xr, yr, margin=remove_nanbefore)
        idx  = np.where(mask)[0]

        time_list.append(t[idx])
        x_left_list.append(xl[idx]);  y_left_list.append(yl[idx])
        x_right_list.append(xr[idx]); y_right_list.append(yr[idx])
        cat_left_list.append(cl[idx]); cat_right_list.append(cr[idx])
        saccade_list.append(sac[idx]); fixation_list.append(fix[idx])

    # Concatenate across all files
    filtered_time          = np.concatenate(time_list)
    filtered_x_left        = np.concatenate(x_left_list)
    filtered_y_left        = np.concatenate(y_left_list)
    filtered_x_right       = np.concatenate(x_right_list)
    filtered_y_right       = np.concatenate(y_right_list)
    filtered_category_left = np.concatenate(cat_left_list)
    filtered_category_right= np.concatenate(cat_right_list)
    filtered_saccade       = np.concatenate(saccade_list)
    filtered_fixation      = np.concatenate(fixation_list)

    print(f"Total samples after NaN filtering: {len(filtered_x_left):,}")
    print(f"Unique fixation labels : {np.unique(filtered_fixation)}")
    print(f"Unique saccade labels  : {np.unique(filtered_saccade)}")

    return dict(
        filtered_time=filtered_time,
        filtered_x_left=filtered_x_left,
        filtered_y_left=filtered_y_left,
        filtered_x_right=filtered_x_right,
        filtered_y_right=filtered_y_right,
        filtered_category_left=filtered_category_left,
        filtered_category_right=filtered_category_right,
        filtered_saccade=filtered_saccade,
        filtered_fixation=filtered_fixation,
    )

# Velocity and angle computations

def compute_velocity(filtered_x_left, filtered_y_left, filtered_time):
    dx = np.diff(filtered_x_left)
    dy = np.diff(filtered_y_left)
    dt = np.diff(filtered_time)
    dt = np.where(dt == 0, 1e-6, dt)
    point_disp = np.sqrt(dx**2 + dy**2)
    point_velo = point_disp / dt
    return point_velo


def compute_effective_velocity(filtered_x_left, filtered_y_left, filtered_time):
    dx = np.diff(filtered_x_left)
    dy = np.diff(filtered_y_left)
    dt = np.diff(filtered_time)
    dt = np.where(dt == 0, 1e-6, dt)
    point_disp = np.sqrt(dx**2 + dy**2)
    point_velo = point_disp / dt

    # Angle between consecutive displacement vectors
    theta_np = []
    for i in range(len(dx) - 1):
        vec1 = [dx[i], dy[i]]
        vec2 = [dx[i + 1], dy[i + 1]]
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            theta_np.append(np.nan)
            continue
        unit1 = vec1 / norm1
        unit2 = vec2 / norm2
        dot_product = np.clip(np.dot(unit1, unit2), -1.0, 1.0)
        theta_np.append(np.abs(np.arccos(dot_product)))

    theta_np = np.array(theta_np)
    valid_theta_indices = ~np.isnan(theta_np)
    theta_valid = theta_np[valid_theta_indices]

    point_velo_eff = np.abs(point_velo[:-1][valid_theta_indices] * np.cos(theta_valid))

    # Align positions to the effective velocity array
    x_val_corr = filtered_x_left[1:-1][valid_theta_indices]
    y_val_corr = filtered_y_left[1:-1][valid_theta_indices]

    return point_velo_eff, x_val_corr, y_val_corr, valid_theta_indices



def binocular_coordination(data):
    xl = data['filtered_x_left'];  xr = data['filtered_x_right']
    yl = data['filtered_y_left'];  yr = data['filtered_y_right']
    cl = data['filtered_category_left']
    cr = data['filtered_category_right']

    x_corr = np.corrcoef(xl, xr)[0, 1]
    y_corr = np.corrcoef(yl, yr)[0, 1]

    fix_agree = np.mean((cl == 1) & (cr == 1)) / max(np.mean(cl == 1), 1e-9) * 100
    sac_agree = np.mean((cl == 2) & (cr == 2)) / max(np.mean(cl == 2), 1e-9) * 100

    print(f"Binocular X correlation : {x_corr:.4f}")
    print(f"Binocular Y correlation : {y_corr:.4f}")
    print(f"Fixation agreement      : {fix_agree:.2f}%")
    print(f"Saccade agreement       : {sac_agree:.2f}%")

    return dict(x_corr=x_corr, y_corr=y_corr,
                fix_agreement=fix_agree, sac_agreement=sac_agree)
