#!/usr/bin/env python3
"""
Adjust MoveIt eye-in-hand calibration output (EEF -> camera_optical) into the mount transform
expected by a URDF joint (EEF -> camera_mount_link), using the camera's fixed frame chain in URDF.

Assumptions:
- MoveIt calibration output in YAML encodes:  fer_link8 -> camera_color_optical_frame
  (translation xyz in meters, and roll/pitch/yaw in radians).
- URDF contains fixed joints that define:       fer_ref_camera_link -> camera_color_optical_frame
- You want to write YAML encoding:              fer_link8 -> fer_ref_camera_link

python3 adjust_calib_yaml.py \
  --urdf fer_franka_hand_with_camera.urdf \
  --in-yaml calibrated_params.yaml \
  --out-yaml calibrated_params_adjusted.yaml \
  --assume-input-is-eef-to-optical

"""

import argparse
import math
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import yaml


def rpy_to_rot(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF/ROS convention: R = Rz(yaw) * Ry(pitch) * Rx(roll)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)

    Rx = np.array([[1, 0, 0],
                   [0, cr, -sr],
                   [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp],
                   [0, 1, 0],
                   [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0],
                   [sy, cy, 0],
                   [0, 0, 1]], dtype=float)

    return Rz @ Ry @ Rx


def rot_to_rpy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of R = Rz(yaw) * Ry(pitch) * Rx(roll). Returns (roll,pitch,yaw)."""
    # pitch = atan2(-r20, sqrt(r00^2 + r10^2))
    r00, r10, r20 = R[0, 0], R[1, 0], R[2, 0]
    pitch = math.atan2(-r20, math.sqrt(r00 * r00 + r10 * r10))

    eps = 1e-9
    if abs(math.cos(pitch)) > eps:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        # Gimbal lock: yaw set to 0, roll absorbs remaining rotation
        roll = math.atan2(-R[1, 2], R[1, 1])
        yaw = 0.0

    return roll, pitch, yaw


def make_T(xyz, rpy) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_rot(*rpy)
    T[:3, 3] = np.array(xyz, dtype=float)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def parse_origin(origin_elem) -> np.ndarray:
    if origin_elem is None:
        return np.eye(4, dtype=float)
    xyz_str = origin_elem.get("xyz", "0 0 0").strip()
    rpy_str = origin_elem.get("rpy", "0 0 0").strip()
    xyz = [float(v) for v in xyz_str.split()]
    rpy = [float(v) for v in rpy_str.split()]
    return make_T(xyz, rpy)


def build_tf_graph_from_urdf(urdf_path: Path) -> dict[str, list[tuple[str, np.ndarray]]]:
    """
    Build an undirected graph of link frames with transforms on edges.
    For each joint: store parent->child and child->parent (inverse).
    """
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    graph: dict[str, list[tuple[str, np.ndarray]]] = {}

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        if parent is None or child is None:
            continue

        parent_link = parent.get("link")
        child_link = child.get("link")
        if not parent_link or not child_link:
            continue

        T_pc = parse_origin(joint.find("origin"))

        graph.setdefault(parent_link, []).append((child_link, T_pc))
        graph.setdefault(child_link, []).append((parent_link, inv_T(T_pc)))

    return graph


def find_transform(graph: dict[str, list[tuple[str, np.ndarray]]], src: str, dst: str) -> np.ndarray:
    """BFS to find any path from src to dst and accumulate transforms along the way."""
    if src == dst:
        return np.eye(4, dtype=float)

    q = deque([src])
    visited = {src}
    # store: node -> (prev_node, T_prev_to_node)
    parent_map: dict[str, tuple[str, np.ndarray]] = {}

    while q:
        u = q.popleft()
        for v, T_uv in graph.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            parent_map[v] = (u, T_uv)
            if v == dst:
                q.clear()
                break
            q.append(v)

    if dst not in parent_map:
        raise RuntimeError(f"No transform path found in URDF graph between '{src}' and '{dst}'.")

    # Reconstruct transform: src -> dst
    T = np.eye(4, dtype=float)
    node = dst
    chain = []
    while node != src:
        prev, T_prev_node = parent_map[node]
        chain.append((prev, node, T_prev_node))
        node = prev
    chain.reverse()

    for _, _, T_uv in chain:
        T = T @ T_uv
    return T


def load_moveit_yaml(yaml_path: Path, section: str) -> dict:
    data = yaml.safe_load(yaml_path.read_text())
    if section not in data or not isinstance(data[section], dict):
        raise RuntimeError(f"Expected a top-level mapping '{section}:' in {yaml_path}")
    return data


def get_xyz_rpy(d: dict, section: str) -> tuple[np.ndarray, np.ndarray]:
    m = d[section]
    xyz = np.array([float(m["x"]), float(m["y"]), float(m["z"])], dtype=float)
    rpy = np.array([float(m["roll"]), float(m["pitch"]), float(m["yaw"])], dtype=float)
    return xyz, rpy


def set_xyz_rpy(d: dict, section: str, xyz: np.ndarray, rpy: np.ndarray) -> None:
    m = d[section]
    m["x"], m["y"], m["z"] = [float(v) for v in xyz.tolist()]
    m["roll"], m["pitch"], m["yaw"] = [float(v) for v in rpy.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, type=Path, help="Full URDF file containing the camera frame chain.")
    ap.add_argument("--in-yaml", required=True, type=Path, help="Input YAML (MoveIt output).")
    ap.add_argument("--out-yaml", required=True, type=Path, help="Output YAML (rewritten for URDF joint).")
    ap.add_argument("--yaml-section", default="camera", help="Top-level key that holds x,y,z,roll,pitch,yaw.")
    ap.add_argument("--eef-frame", default="fer_link8", help="EEF link frame used in calibration.")
    ap.add_argument("--optical-frame", default="camera_color_optical_frame", help="Camera optical frame.")
    ap.add_argument("--mount-frame", default="fer_ref_camera_link", help="URDF mount link frame (child of EEF joint).")
    ap.add_argument("--assume-input-is-eef-to-optical", action="store_true",
                    help="If set, interpret input YAML as EEF->Optical. (Recommended)")
    ap.add_argument("--assume-input-is-optical-to-eef", action="store_true",
                    help="If set, interpret input YAML as Optical->EEF (will invert).")
    args = ap.parse_args()

    if args.assume_input_is_optical_to_eef and args.assume_input_is_eef_to_optical:
        raise RuntimeError("Choose only one of --assume-input-is-eef-to-optical / --assume-input-is-optical-to-eef")

    # Default behavior: EEF->Optical
    optical_to_eef = bool(args.assume_input_is_optical_to_eef)

    # 1) Load MoveIt output transform from YAML
    d = load_moveit_yaml(args.in_yaml, args.yaml_section)
    xyz_in, rpy_in = get_xyz_rpy(d, args.yaml_section)
    T_in = make_T(xyz_in, rpy_in)

    if optical_to_eef:
        T_eef_opt = inv_T(T_in)
    else:
        T_eef_opt = T_in

    # 2) From URDF: compute T_mount_opt  (mount-frame -> optical-frame)
    graph = build_tf_graph_from_urdf(args.urdf)
    T_mount_opt = find_transform(graph, args.mount_frame, args.optical_frame)

    # 3) Compute mount transform expected by URDF joint: T_eef_mount
    #    T_eef_opt = T_eef_mount * T_mount_opt  =>  T_eef_mount = T_eef_opt * inv(T_mount_opt)
    T_eef_mount = T_eef_opt @ inv_T(T_mount_opt)

    # 4) Write YAML with EEF->mount
    xyz_out = T_eef_mount[:3, 3]
    roll, pitch, yaw = rot_to_rpy(T_eef_mount[:3, :3])
    rpy_out = np.array([roll, pitch, yaw], dtype=float)

    set_xyz_rpy(d, args.yaml_section, xyz_out, rpy_out)

    # Optional: keep/update parent_frame field if present
    if "parent_frame" in d[args.yaml_section]:
        d[args.yaml_section]["parent_frame"] = args.eef_frame

    args.out_yaml.write_text(yaml.safe_dump(d, sort_keys=False))
    print("Wrote:", args.out_yaml)
    print("Meaning: {} -> {}".format(args.eef_frame, args.mount_frame))


if __name__ == "__main__":
    main()
