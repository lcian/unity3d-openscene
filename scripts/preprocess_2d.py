#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import os

from tqdm import tqdm
from imageio import imwrite
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    MeshRasterizer,
    MeshRenderer,
    MeshRendererWithFragments,
    PointLights,
    RasterizationSettings,
    SoftPhongShader,
    look_at_view_transform,
)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

DEFAULT_IMAGE_SIZE = (640, 360)  # same as Replica
DEFAULT_TEXTURE_ATLAS_SIZE = 32  # requires GPU
DEFAULT_NUM_VIEWS = 100

BATCH_SIZE = 20

FOCAL_LENGTH = 2.0


# .obj to 2D views in a format equivalent to the one expected by OpenScene for Replica
def obj_to_views(
    in_dir,
    out_dir,
    image_size=DEFAULT_IMAGE_SIZE,
    texture_atlas_size=DEFAULT_TEXTURE_ATLAS_SIZE,
    num_views=DEFAULT_NUM_VIEWS,
):

    assert num_views % BATCH_SIZE == 0

    image_size = (image_size[1], image_size[0])  # :)

    in_file = os.path.join(in_dir, "mesh.obj")
    mesh_name = in_dir.split("/")[-1]
    out_dir = os.path.abspath(out_dir)
    scene_out_dir = os.path.join(out_dir, mesh_name)
    color_dir = os.path.join(scene_out_dir, "color")
    depth_dir = os.path.join(scene_out_dir, "depth")
    pose_dir = os.path.join(scene_out_dir, "pose")
    os.makedirs(scene_out_dir, exist_ok=True)
    os.makedirs(color_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(pose_dir, exist_ok=True)

    mesh = load_objs_as_meshes(
        [in_file],
        load_textures=True,
        create_texture_atlas=True,
        texture_atlas_size=texture_atlas_size,
        device=device,
    )

    # center the mesh
    verts = mesh.verts_packed()
    center = verts.mean(0)
    mesh.offset_verts_(-center)

    # generate viewing angles and cameras
    elev = torch.zeros(num_views)
    azim = torch.linspace(0, 360, num_views)

    for batch in tqdm(range(num_views // BATCH_SIZE)):
        batch_i = batch * BATCH_SIZE
        batch_j = (batch + 1) * BATCH_SIZE
        R, T = look_at_view_transform(
            dist=0.1, elev=elev[batch_i:batch_j], azim=azim[batch_i:batch_j]
        )
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=FOCAL_LENGTH)

        # arbitrarily choose one particular view to be used for rasterizer and shader
        camera = PerspectiveCameras(
            device=device,
            R=R[None, 1, ...],
            T=T[None, 1, ...],
            focal_length=FOCAL_LENGTH,
        )

        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
        )

        renderer = MeshRendererWithFragments(
            rasterizer=MeshRasterizer(cameras=camera, raster_settings=raster_settings),
            shader=SoftPhongShader(
                device=device,
                cameras=camera,
            ),
        )

        # render
        meshes = mesh.extend(BATCH_SIZE)
        rendering, fragments = renderer(meshes, cameras=cameras)
        rendering_rgb = [rendering[i, ..., :3] for i in range(BATCH_SIZE)]
        zbufs = [fragments.zbuf[i] for i in range(BATCH_SIZE)]

        # color views
        transform = transforms.ToPILImage()
        for i, rgb in enumerate(rendering_rgb):
            img = transform(rgb.permute(2, 0, 1))
            img.save(os.path.join(color_dir, f"{(batch * BATCH_SIZE) + i}.jpg"))

        # depth views for i, depth in enumerate(zbufs):
        for i, depth in enumerate(zbufs):
            depth = depth.squeeze()
            depth = (depth - depth.min()) / (depth.max() - depth.min())
            depth = depth * 65535
            depth = depth.cpu().numpy().astype(np.uint16)
            imwrite(os.path.join(depth_dir, f"{(batch * BATCH_SIZE) + i}.png"), depth)

        # camera intrinsic (all same)
        intrinsic = torch.eye(4)
        intrinsic[0][0] = FOCAL_LENGTH * 1000
        intrinsic[1][1] = FOCAL_LENGTH * 1000
        intrinsic[0][2] = image_size[1] / 2
        intrinsic[1][2] = image_size[0] / 2
        np.savetxt(
            os.path.join(out_dir, "intrinsics.txt"),
            intrinsic.cpu().numpy(),
            fmt="%1.18e",
            delimiter=" ",
        )

        # camera poses
        poses = [
            torch.linalg.inv(camera.get_world_to_view_transform().get_matrix())
            for camera in cameras
        ]
        poses = torch.cat(poses)
        poses.reshape(1, len(poses) * 4, 4)
        for i, pose in enumerate(poses):
            np.savetxt(
                os.path.join(pose_dir, f"{(batch * BATCH_SIZE) + i}.txt"),
                pose.cpu().numpy(),
                fmt="%1.18e",
                delimiter=" ",
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate 2D views from .obj mesh")
    parser.add_argument("in_dir", type=str, help="Input directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
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
    obj_to_views(
        args.in_dir,
        args.out_dir,
        (args.views_w, args.views_h),
        args.texture_atlas_size,
        args.num_views,
    )
