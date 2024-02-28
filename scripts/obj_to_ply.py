#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import os

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


def obj_to_ply(filename, num_samples=2_000_000, ascii=False, create_meshlab=False):

    ply_filename = filename.replace(".obj", ".ply")

    mesh = load_objs_as_meshes(
        [filename], load_textures=True, create_texture_atlas=True
    )
    points, features = sample_points_from_meshes(
        mesh, num_samples=num_samples, return_textures=True
    )
    pointcloud = Pointclouds(points=points, features=features)
    IO().save_pointcloud(pointcloud, ply_filename, binary=not ascii)

    if create_meshlab:
        features = (features * 255).to(torch.uint8)
        pointcloud = Pointclouds(points=points, features=features)
        IO().save_pointcloud(pointcloud, ply_filename + ".tmp.ply")

        ply = PlyData.read(ply_filename + ".tmp.ply")
        verts = ply["vertex"]
        verts.ply_property("red").val_dtype = "u1"
        verts.ply_property("green").val_dtype = "u1"
        verts.ply_property("blue").val_dtype = "u1"
        PlyData(ply, text=ascii).write(
            "".join(ply_filename.split(".")[:-1]) + ".meshlab.ply"
        )

        os.remove(ply_filename + ".tmp.ply")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate .ply file from .obj")
    parser.add_argument("filename", type=str, help="Input file")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=2_000_000,
        help="Number of samples to take from the mesh",
    )
    parser.add_argument(
        "--ascii", action="store_true", help="Save as ASCII instead of binary"
    )
    parser.add_argument(
        "--create_meshlab",
        action="store_true",
        help="Create an additional .ply file compatible with Meshlab",
    )
    args = parser.parse_args()
    obj_to_ply(args.filename, args.num_samples, args.ascii, args.create_meshlab)
