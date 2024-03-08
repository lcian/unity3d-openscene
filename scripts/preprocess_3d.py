#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import os

import numpy as np
import torch
from plyfile import PlyData
from pytorch3d.io import IO, load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Pointclouds

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

DEFAULT_NUM_SAMPLES = 100_000


# .obj to .pth in a format equivalent to the one expected by OpenScene for Replica
def obj_to_pth(in_dir, out_dir, num_samples=DEFAULT_NUM_SAMPLES):

    in_file = os.path.join(in_dir, "mesh.obj")
    mesh_name = in_dir.split("/")[-1]
    out_file = os.path.join(out_dir, mesh_name + ".pth")
    tmp_out_file = os.path.join(out_dir, mesh_name + ".tmp.ply")

    mesh = load_objs_as_meshes([in_file], load_textures=True, create_texture_atlas=True)

    # center the mesh
    verts = mesh.verts_packed()
    center = verts.mean(0)
    mesh.offset_verts_(-center)

    points, features = sample_points_from_meshes(
        mesh, num_samples=num_samples, return_textures=True
    )
    pointcloud = Pointclouds(points=points, features=features)
    IO().save_pointcloud(pointcloud, tmp_out_file, binary=True)
    ply = PlyData.read(tmp_out_file)

    # from openscene/scripts/preprocess/preprocess_3d_replica.py
    verts = np.array([list(x) for x in ply.elements[0]])
    coords = np.ascontiguousarray(verts[:, :3])
    colors = np.ascontiguousarray(verts[:, -3:]) / 127.5 - 1
    labels = 255 * np.ones((coords.shape[0],), dtype=np.int32)
    torch.save((coords, colors, labels), out_file)

    os.remove(tmp_out_file)
    return


# .obj to human-readable .ply usable in Meshlab
def obj_to_ply(in_dir, out_dir, num_samples=DEFAULT_NUM_SAMPLES):

    in_file = os.path.join(in_dir, "mesh.obj")
    mesh_name = in_dir.split("/")[-1]
    out_file = os.path.join(out_dir, mesh_name + ".ply")
    tmp_out_file = os.path.join(out_dir, mesh_name + ".tmp.ply")

    mesh = load_objs_as_meshes([in_file], load_textures=True, create_texture_atlas=True)
    points, features = sample_points_from_meshes(
        mesh, num_samples=num_samples, return_textures=True
    )
    features = (features * 255).to(torch.uint8)
    pointcloud = Pointclouds(points=points, features=features)
    IO().save_pointcloud(pointcloud, tmp_out_file, binary=True)

    ply = PlyData.read(tmp_out_file)
    verts = ply["vertex"]
    verts.ply_property("red").val_dtype = "u1"
    verts.ply_property("green").val_dtype = "u1"
    verts.ply_property("blue").val_dtype = "u1"
    PlyData(ply, text=True).write(out_file)

    os.remove(tmp_out_file)
    return


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate .pth/.ply file from .obj mesh"
    )
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help="Number of samples to take from the mesh",
    )
    parser.add_argument(
        "--ply",
        action="store_true",
        help="Save as human-readable .ply usable in Meshlab instead of .pth",
    )
    args = parser.parse_args()
    if args.ply:
        obj_to_ply(args.in_dir, args.out_dir, args.num_samples)
    else:
        obj_to_pth(args.in_dir, args.out_dir, args.num_samples)
