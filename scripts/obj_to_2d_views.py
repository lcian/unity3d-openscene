#!/usr/bin/env python3
# pyright: reportGeneralTypeIssues=false
import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    SoftPhongShader,
)
import torchvision.transforms as transforms

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")


# from https://pytorch3d.org/tutorials/fit_textured_mesh
# TODO: camera positioning
def obj_to_2d_views(filename, image_size=256, num_views=20):

    mesh = load_objs_as_meshes([filename], load_textures=True, create_texture_atlas=True, device=device)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
    # to its original center and scale.  Note that normalizing the target mesh, 
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center)
    mesh.scale_verts_((1.0 / float(scale)))

    # Get a batch of viewing angles. 
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)

    # Place a point light in front of the object.
    #lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    R, T = look_at_view_transform(dist=3.5, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    # We arbitrarily choose one particular view that will be used to visualize results
    camera = FoVPerspectiveCameras(device=device, R=R[None, 1, ...], 
                                      T=T[None, 1, ...]) 

    raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    # Create a Phong renderer by composing a rasterizer and a shader. The textured 
    # Phong shader will interpolate the texture uv coordinates for each vertex, 
    # sample from a texture image and apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=camera,
            #lights=lights
        )
    )

    # Create a batch of meshes by repeating the cow mesh and associated textures. 
    # Meshes has a useful `extend` method which allows us do this very easily. 
    # This also extends the textures. 
    meshes = mesh.extend(num_views)

    # Render the cow mesh from each viewing angle
    target_images = renderer(meshes, cameras=cameras)#, lights=lights)

    # Our multi-view cow dataset will be represented by these 2 lists of tensors,
    # each of length num_views.
    target_rgb = [target_images[i, ..., :3] for i in range(num_views)]
    #target_cameras = [FoVPerspectiveCameras(device=device, R=R[None, i, ...], T=T[None, i, ...]) for i in range(num_views)]
    
    transform = transforms.ToPILImage()
    for i, rgb in enumerate(target_rgb):
        img = transform(rgb.permute(2, 0, 1))
        img.save(filename.replace(".obj", "") + f"_view_{i}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate 2D views from 3D .obj')
    parser.add_argument('filename', type=str, help='Input file')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    parser.add_argument('--num_views', type=int, default=20, help='Number of views')
    args = parser.parse_args()
    obj_to_2d_views(args.filename, args.image_size, args.num_views)
