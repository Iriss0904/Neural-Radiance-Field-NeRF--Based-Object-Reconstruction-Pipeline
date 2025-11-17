import os
import argparse
import numpy as np
import trimesh
import pyrender
from PIL import Image
import imageio

def render_turntable(mesh_path, out_dir, video_path, frames=180, img_size=512):
    # -----------------------------
    # Load mesh
    # -----------------------------
    print(f"[Load] Mesh from: {mesh_path}")
    mesh_trimesh = trimesh.load(mesh_path)
    if not isinstance(mesh_trimesh, trimesh.Trimesh):
        mesh_trimesh = mesh_trimesh.dump(concatenate=True)

    # -----------------------------
    # Convert to pyrender mesh
    # -----------------------------
    mesh = pyrender.Mesh.from_trimesh(mesh_trimesh, smooth=True)

    # -----------------------------
    # Create scene
    # -----------------------------
    scene = pyrender.Scene()
    scene.add(mesh)

    # Set up camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0)
    camera_node = scene.add(camera, pose=np.eye(4))

    # Lighting
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=5.0)
    scene.add(light, pose=np.eye(4))

    # -----------------------------
    # Create renderer
    # -----------------------------
    r = pyrender.OffscreenRenderer(img_size, img_size)

    # -----------------------------
    # Prepare output dir
    # -----------------------------
    os.makedirs(out_dir, exist_ok=True)
    print(f"[Render] Generating {frames} frames...")

    # -----------------------------
    # Turntable rendering
    # -----------------------------
    angles = np.linspace(0, 2*np.pi, frames)
    rendered_frames = []

    for idx, theta in enumerate(angles):
        # Camera rotation around Y axis
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = [
            [ np.cos(theta), 0, np.sin(theta)],
            [ 0,            1, 0            ],
            [-np.sin(theta), 0, np.cos(theta)]
        ]
        cam_pose[0, 3] = 1.8      # distance X
        cam_pose[1, 3] = 1.0      # distance Y
        cam_pose[2, 3] = 1.8      # distance Z

        scene.set_pose(camera_node, pose=cam_pose)

        # Render
        color, depth = r.render(scene)
        frame_path = os.path.join(out_dir, f"frame_{idx:04d}.png")
        Image.fromarray(color).save(frame_path)
        rendered_frames.append(color)

        print(f"  Frame {idx+1}/{frames}")

    # -----------------------------
    # Save video
    # -----------------------------
    print("[Video] Writing mp4...")
    imageio.mimsave(video_path, rendered_frames, fps=30)
    print(f"[Done] Video saved to: {video_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True, help="input mesh file")
    parser.add_argument("--out_dir", required=True, help="directory for frames")
    parser.add_argument("--video", required=True, help="output mp4 path")
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()

    render_turntable(
        mesh_path=args.mesh,
        out_dir=args.out_dir,
        video_path=args.video,
        frames=args.frames,
    )
