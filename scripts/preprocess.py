#!/usr/bin/env python3
import os

import preprocess_2d
import preprocess_3d

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess .obj + .mtl scenes")
    parser.add_argument(
        "in_dir",
        type=str,
        help="Input directory, containing a subdirectory for each scene",
    )
    parser.add_argument("out_dir", type=str, help="Output directory")

    # arguments for 3D preprocessing
    parser.add_argument(
        "--num_samples",
        type=int,
        default=preprocess_3d.DEFAULT_NUM_SAMPLES,
        help="Number of samples to take from the mesh",
    )

    # arguments for 2D preprocessing
    parser.add_argument(
        "--views_w",
        type=int,
        default=preprocess_2d.DEFAULT_IMAGE_SIZE[0],
        help="View width",
    )
    parser.add_argument(
        "--views_h",
        type=int,
        default=preprocess_2d.DEFAULT_IMAGE_SIZE[1],
        help="View height",
    )
    parser.add_argument(
        "--texture_atlas_size",
        type=int,
        default=preprocess_2d.DEFAULT_TEXTURE_ATLAS_SIZE,
        help="View height",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=preprocess_2d.DEFAULT_NUM_VIEWS,
        help="Number of views",
    )

    args = parser.parse_args()

    two_d_out_dir = os.path.join(args.out_dir, "2d")
    os.makedirs(two_d_out_dir, exist_ok=True)
    three_d_out_dir = os.path.join(args.out_dir, "3d")
    os.makedirs(three_d_out_dir, exist_ok=True)
    for scene_dir in os.listdir(args.in_dir):
        scene_in_dir = os.path.join(args.in_dir, scene_dir)

        preprocess_3d.obj_to_pth(scene_in_dir, three_d_out_dir, args.num_samples)

        scene_out_dir = os.path.join(two_d_out_dir, scene_dir)
        os.makedirs(scene_out_dir, exist_ok=True)
        preprocess_2d.obj_to_views(
            scene_in_dir,
            scene_out_dir,
            (args.views_w, args.views_h),
            args.texture_atlas_size,
            args.num_views,
        )
