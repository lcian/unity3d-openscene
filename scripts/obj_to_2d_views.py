#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import os

import numpy as np
import torch
import torchvision.transforms as transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    MeshRasterizer,
    MeshRenderer,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    SoftPhongShader,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

DEFAULT_IMAGE_SIZE = (1200, 680)  # same as Replica
DEFAULT_TEXTURE_ATLAS_SIZE = 64  # works well for the examples, requires GPU
DEFAULT_NUM_VIEWS = 32


# from https://pytorch3d.org/tutorials/fit_textured_mesh
def obj_to_2d_views(
    filename,
    image_size=DEFAULT_IMAGE_SIZE,
    texture_atlas_size=DEFAULT_TEXTURE_ATLAS_SIZE,
    num_views=DEFAULT_NUM_VIEWS,
):

    mesh = load_objs_as_meshes(
        [filename],
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=texture_atlas_size,
        device=device,
    )

    # center the mesh
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    mesh.offset_verts_(-center)

    # generate viewing angles and cameras
    elev = torch.zeros(num_views)
    azim = torch.linspace(0, 360, num_views)
    R, T = look_at_view_transform(dist=0.1, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # arbitrarily choose one particular view to be used for rasterizer and shader
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], T=T[None, 1, ...])

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
        shader=SoftPhongShader(
            device=device,
            cameras=camera,
        ),
    )

    # render
    meshes = mesh.extend(num_views)
    target_images = renderer(meshes, cameras=cameras)
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]

    # save
    transform = transforms.ToPILImage()
    for i, rgb in enumerate(target_rgb):
        img = transform(rgb.permute(2, 0, 1))
        img.save(filename.replace(".obj", "") + f"_view_{i}.png")

    directory = "/".join(filename.split("/")[:-1])

    # save camera intrinsic
    intrinsic = cameras[0].get_projection_transform().get_matrix()
    intrinsic = intrinsic[0].cpu().numpy()
    np.savetxt(
        os.path.join(directory, "intrinsics.txt"),
        intrinsic,
        fmt="%1.18e",
        delimiter=" ",
    )

    # save camera poses
    poses = [camera.get_world_to_view_transform().get_matrix() for camera in cameras]
    poses = torch.cat(poses)
    poses.reshape(1, len(poses) * 4, 4)
    poses = poses[0].cpu().numpy()
    np.savetxt(os.path.join(directory, "traj.txt"), poses, fmt="%1.18e", delimiter=" ")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2D views from 3D .obj")
    parser.add_argument("filename", type=str, help="Input file")
    parser.add_argument(
        "--views_w", type=int, default=DEFAULT_IMAGE_SIZE[0], help="View width"
    )
    parser.add_argument(
        "--views_h", type=int, default=DEFAULT_IMAGE_SIZE[1], help="View height"
    )
    parser.add_argument(
        "--texture_atlas_size",
        type=int,
        default=DEFAULT_TEXTURE_ATLAS_SIZE,
        help="View height",
    )
    parser.add_argument(
        "--num_views", type=int, default=DEFAULT_NUM_VIEWS, help="Number of views"
    )
    args = parser.parse_args()
    obj_to_2d_views(
        args.filename,
        (args.views_w, args.views_h),
        args.texture_atlas_size,
        args.num_views,
    )
