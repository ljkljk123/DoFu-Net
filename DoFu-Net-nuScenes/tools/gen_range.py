import os
import math
import pickle
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from tools.utils import eulerAngles2rotationMat, check_dir, load_config


def py_top3_farthest_mean(points_xyz: np.ndarray,
                          proj_H: int = 32,
                          proj_W: int = 768,
                          max_radius: float = 80.0,
                          pitch_mode: str = "minmax",
                          pitch_percentiles=(1.0, 99.0)):
    """
    Dome-style P-Y image with FIXED bins (proj_H, proj_W) and ADAPTIVE per-frame pitch FoV.

    Per (pitch_bin, yaw_bin), we follow your Dome C++ logic:
      1) collect abs_depth = sqrt(x^2+y^2+z^2)
      2) sort depths ascending
      3) avg_depth = mean of farthest 3 depths (largest 3). If <3 points, mean of all.
      4) weight = log(count + 1)
      5) intensity = avg_depth * weight
      6) pixel = intensity / max_radius * 255

    Output:
      py_img: float32 (H,W), default 0 for empty bins (matches Dome behavior more than -1)
      pitch_minmax: (pitch_min, pitch_max) in radians for this frame
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    abs_depth = np.linalg.norm(points_xyz, axis=1)
    valid = (abs_depth > 1e-6) & (abs_depth < max_radius)

    py_img = np.full((proj_H, proj_W), -1,dtype=np.float32)
    if not np.any(valid):
        return py_img, (-0.5, 0.5)

    x, y, z, abs_depth = x[valid], y[valid], z[valid], abs_depth[valid]

    # bins
    yaw = np.arctan2(y, x)  # [-pi, pi] in radians
    # map yaw to [0, proj_W)
    yaw_bin = yaw * 180.0 / np.pi
    yaw_bin = np.where(yaw_bin < 0, yaw_bin + 360.0, yaw_bin)
    yaw_bin = np.floor(yaw_bin / 360.0 * proj_W).astype(np.int32)
    yaw_bin = np.clip(yaw_bin, 0, proj_W - 1)

    xy = np.sqrt(x * x + y * y)
    pitch = np.arctan2(z, np.maximum(xy, 1e-6))  # radians

    if pitch_mode == "minmax":
        pitch_min = float(np.min(pitch))
        pitch_max = float(np.max(pitch))
    elif pitch_mode == "percentile":
        lo, hi = pitch_percentiles
        pitch_min = float(np.percentile(pitch, lo))
        pitch_max = float(np.percentile(pitch, hi))
    else:
        raise ValueError("pitch_mode must be 'minmax' or 'percentile'")

    if abs(pitch_max - pitch_min) < 1e-6:
        pitch_min -= 1e-3
        pitch_max += 1e-3

    # map pitch to [0, proj_H)
    pitch_bin = (pitch - pitch_min) / (pitch_max - pitch_min)  # [0,1]
    pitch_bin = np.floor(pitch_bin * proj_H).astype(np.int32)
    pitch_bin = np.clip(pitch_bin, 0, proj_H - 1)

    # aggregate depths per bin
    # We'll build a dict of lists (simple + faithful to C++; generation is offline anyway)
    bin_depths = {}
    for pb, yb, d in zip(pitch_bin, yaw_bin, abs_depth):
        key = (pb, yb)
        if key not in bin_depths:
            bin_depths[key] = []
        bin_depths[key].append(float(d))

    for (pb, yb), depths in bin_depths.items():
        depths.sort()  # ascending
        if len(depths) >= 3:
            avg_depth = (depths[-1] + depths[-2] + depths[-3]) / 3.0
        else:
            avg_depth = sum(depths) / float(len(depths))

        weight = math.log(len(depths) + 1.0)
        intensity = avg_depth * weight
        py_img[pb, yb] = intensity
        # nearest = depths[0]
        # py_img[pb, yb] = nearest

    return py_img.astype(np.float32), (pitch_min, pitch_max)

#3) project_ry：生成 R-Y（RangeXY-Yaw）视图，值是每格最大的 z
def project_ry_maxz(points_xyz: np.ndarray, proj_R: int, proj_W: int, max_range_xy: float = 80.0):
    """
    R-Y: FIXED bins, value=max z in each bin (empty -> 0).
    Shape: (proj_R, proj_W)
    """
    x = points_xyz[:, 0]
    y = points_xyz[:, 1]
    z = points_xyz[:, 2]

    r_xy = np.sqrt(x * x + y * y)
    valid = (r_xy > 1e-6) & (r_xy < max_range_xy)

    ry = np.full((proj_R, proj_W),-1, dtype=np.float32)
    if not np.any(valid):
        return ry

    x, y, z, r_xy = x[valid], y[valid], z[valid], r_xy[valid]
    yaw = np.arctan2(y, x)
    yaw_deg = yaw * 180.0 / np.pi
    yaw_deg = np.where(yaw_deg < 0, yaw_deg + 360.0, yaw_deg)
    yaw_bin = np.floor(yaw_deg / 360.0 * proj_W).astype(np.int32)
    yaw_bin = np.clip(yaw_bin, 0, proj_W - 1)

    r_bin = np.floor((r_xy / max_range_xy) * proj_R).astype(np.int32)
    r_bin = np.clip(r_bin, 0, proj_R - 1)

    # max z
    for rb, yb, zz in zip(r_bin, yaw_bin, z):
        if zz > ry[rb, yb]:
            ry[rb, yb] = float(zz)
    return ry

#4) project_rp_density：生成 R-P（Range-Pitch）视图，值是点密度（计数）
def project_rp_density(points_xyz: np.ndarray, proj_R: int, proj_H: int, pitch_minmax, max_range_3d: float = 80.0):
    """
    R-P density: FIXED bins, uses SAME adaptive pitch_min/pitch_max as PY for consistency.
    Shape: (proj_R, proj_H)
    """
    # np.linalg.norm:计算向量的范数，其中axis表示按照行还是列进行计算，axis=1表示按照行进行计算
    # 在这里表示计算每个点到原点的距离
    depth = np.linalg.norm(points_xyz, axis=1)
    valid = (depth > 1e-6) & (depth < max_range_3d) #布尔数组

    rp = np.full((proj_R, proj_H),-1, dtype=np.float32)
    if not np.any(valid):
        return rp

    pts = points_xyz[valid] # 3. 用布尔数组过滤points_xyz
    depth = depth[valid]  # 4. 用同样的布尔数组过滤depth

    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xy = np.sqrt(x * x + y * y)
    pitch = np.arctan2(z, np.maximum(xy, 1e-6))

    pitch_min, pitch_max = pitch_minmax
    if abs(pitch_max - pitch_min) < 1e-6:
        pitch_min -= 1e-3
        pitch_max += 1e-3

    p_bin = (pitch - pitch_min) / (pitch_max - pitch_min)  # [0,1]
    p_bin = np.floor(p_bin * proj_H).astype(np.int32)
    p_bin = np.clip(p_bin, 0, proj_H - 1)

    r_bin = np.floor((depth / max_range_3d) * proj_R).astype(np.int32)
    r_bin = np.clip(r_bin, 0, proj_R - 1)
    # zip() 返回的是一个可迭代的zip对象
    for rb, pb in zip(r_bin, p_bin):
        rp[rb, pb] += 1.0
    return rp


def gen_3view_dofu_bins_adaptive_fov(dataroot, infos, target_folder,
                                    proj_H=32, proj_W=768, proj_R=256,
                                    max_radius=80.0,
                                    pitch_mode="minmax",
                                    pitch_percentiles=(1.0, 99.0)):
    """
    Save per frame: .npz with keys py, ry, rp, pitch_min, pitch_max
    - PY matches Dome: farthest-3 mean depth + log(count+1) weight, scaled to 0..255-ish
    - bins follow the configured DoFu-Net view dimensions: yaw=proj_W, pitch=proj_H
    """
    assert os.path.exists(target_folder)
    #主循环：逐帧处理
    for i in range(len(infos)):
        fname = infos[i]['lidar_infos']['LIDAR_TOP']['filename']
        #从 infos 中取出该帧 LIDAR_TOP 的点云文件相对路径。
        pcl = LidarPointCloud.from_file(os.path.join(dataroot, fname))
        # 删除半径 2m 内的近点（通常是车体附近噪声/无意义点）。
        pcl.remove_close(radius=2)

        points = pcl.points[:3, :]  # 3 x N
        rot_mat = eulerAngles2rotationMat(angles=[0, 0, -90])
        points = rot_mat.dot(points).T  # N x 3

        py, pitch_minmax = py_top3_farthest_mean(
            points, proj_H=proj_H, proj_W=proj_W, max_radius=max_radius,
            pitch_mode=pitch_mode, pitch_percentiles=pitch_percentiles
        )
        ry = project_ry_maxz(points, proj_R=proj_R, proj_W=proj_W, max_range_xy=max_radius)
        rp = project_rp_density(points, proj_R=proj_R, proj_H=proj_H, pitch_minmax=pitch_minmax, max_range_3d=max_radius)

        dst_path = os.path.join(target_folder, os.path.split(fname.split('.')[0])[-1]) + '.npz'
        np.savez_compressed(
            dst_path,
            py=py.astype(np.float32),
            ry=ry.astype(np.float32),
            rp=rp.astype(np.float32),
            pitch_min=np.float32(pitch_minmax[0]),
            pitch_max=np.float32(pitch_minmax[1]),
        )
        print('finished:', dst_path)


def main():
    # ===== DoFu-Net view bins =====
    """
   参数	              含义	            对应视图
   proj_H = 32	      pitch方向bin数	PY、RP
   proj_W = 768	      yaw方向bin数	    PY、RY
   proj_R = 64	      range方向bin数	RY、RP
   max_radius = 80.0  最大有效距离（米） 所有视图
    """
    proj_H = 32
    proj_W = 768
    proj_R = 64
    max_radius = 80.0

    # Adaptive FoV per frame (covers all points in current frame)
    pitch_mode = "minmax"          # matches your preference
    pitch_percentiles = (1.0, 99.0)

    cfg = load_config('config/config.yaml')
    dataroot = cfg['data']['data_root_dir']
    target_folder = os.path.join(dataroot, cfg['data'].get('lidar_view_dir', 'samples/RANGE_DOFU_3VIEW'))
    check_dir(target_folder)

    for pkl_name in ['nuscenes_infos-bs.pkl', 'nuscenes_infos-sq.pkl','nuscenes_infos-son.pkl','nuscenes_infos-shv.pkl']:
        with open(os.path.join(dataroot, pkl_name), 'rb') as f:
            infos = pickle.load(f)
        gen_3view_dofu_bins_adaptive_fov(
            dataroot, infos, target_folder,
            proj_H=proj_H, proj_W=proj_W, proj_R=proj_R,
            max_radius=max_radius,
            pitch_mode=pitch_mode,
            pitch_percentiles=pitch_percentiles
        )


if __name__ == '__main__':
    main()
