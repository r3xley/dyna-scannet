#!/usr/bin/env python3
"""
Visualize ScanNet scene data.

Supports multiple formats:
- ScanNet mesh files (.ply)
- Point clouds
- COLMAP reconstructions
- ShapeR pickle files
- Images and camera poses
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not available. Install with: pip install trimesh")

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning: open3d not available. Install with: pip install open3d")

try:
    import rerun as rr
    RERUN_AVAILABLE = True
except ImportError:
    RERUN_AVAILABLE = False
    print("Warning: rerun-sdk not available. Install with: pip install rerun-sdk")


def load_mesh_ply(file_path: Path):
    """Load mesh from PLY file."""
    if TRIMESH_AVAILABLE:
        mesh = trimesh.load(str(file_path))
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Try to get vertex colors if available
        vertex_colors = None
        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
            vertex_colors = mesh.visual.vertex_colors[:, :3]  # RGB only
            if vertex_colors.max() <= 1.0:
                vertex_colors = (vertex_colors * 255).astype(np.uint8)
            else:
                vertex_colors = vertex_colors.astype(np.uint8)
        elif hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'main_color'):
            # Use material color
            color = mesh.visual.material.main_color[:3]
            if color.max() <= 1.0:
                color = (color * 255).astype(np.uint8)
            else:
                color = color.astype(np.uint8)
            vertex_colors = np.tile(color, (len(vertices), 1))
        
        return vertices, faces, vertex_colors
    else:
        raise ImportError("trimesh is required to load PLY files")


def load_point_cloud(file_path: Path):
    """Load point cloud from file."""
    if file_path.suffix == '.ply':
        if TRIMESH_AVAILABLE:
            pc = trimesh.load(str(file_path))
            if hasattr(pc, 'vertices'):
                return pc.vertices
            else:
                return np.array(pc.vertices)
        else:
            raise ImportError("trimesh is required to load PLY files")
    else:
        raise ValueError(f"Unsupported point cloud format: {file_path.suffix}")


def load_shaper_pickle(pickle_path: Path):
    """Load ShapeR pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_colmap_data(data_dir: Path, camera_source: str = 'iphone'):
    """Load COLMAP reconstruction data."""
    colmap_dir = data_dir / camera_source / 'colmap'
    
    cameras = {}
    images = {}
    points3d = {}
    
    # Load cameras
    cameras_path = colmap_dir / 'cameras.txt'
    if cameras_path.exists():
        with open(cameras_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 4:
                    cam_id = int(parts[0])
                    cameras[cam_id] = {
                        'model': parts[1],
                        'width': int(parts[2]),
                        'height': int(parts[3]),
                        'params': [float(p) for p in parts[4:]]
                    }
    
    # Load images
    images_path = colmap_dir / 'images.txt'
    if images_path.exists():
        with open(images_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('#'):
                    i += 1
                    continue
                parts = line.split()
                if len(parts) >= 10:
                    img_id = int(parts[0])
                    qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                    cam_id = int(parts[8])
                    name = parts[9]
                    
                    # Convert quaternion to rotation matrix
                    R = np.array([
                        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
                        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
                        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
                    ])
                    T = np.eye(4)
                    T[:3, :3] = R
                    T[:3, 3] = [tx, ty, tz]
                    
                    images[img_id] = {
                        'name': name,
                        'cam_id': cam_id,
                        'pose': T
                    }
                i += 1
    
    # Load 3D points
    points_path = colmap_dir / 'points3D.txt'
    if points_path.exists():
        with open(points_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) >= 7:
                    point_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    r, g, b = int(parts[4]), int(parts[5]), int(parts[6])
                    points3d[point_id] = {
                        'position': np.array([x, y, z]),
                        'color': np.array([r, g, b]) / 255.0
                    }
    
    return cameras, images, points3d


def visualize_mesh_matplotlib(vertices: np.ndarray, faces: np.ndarray, 
                               title: str = "Mesh", save_path: Optional[Path] = None):
    """Visualize mesh using matplotlib."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot mesh faces
    if len(faces) > 0:
        for face in faces[:1000]:  # Limit for performance
            if len(face) >= 3:
                triangle = vertices[face[:3]]
                ax.add_collection3d(Poly3DCollection([triangle], alpha=0.3, facecolor='cyan', edgecolor='blue'))
    
    # Plot vertices as points
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c='red', s=1, alpha=0.5, label='Vertices')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([vertices[:, 0].max() - vertices[:, 0].min(),
                          vertices[:, 1].max() - vertices[:, 1].min(),
                          vertices[:, 2].max() - vertices[:, 2].min()]).max() / 2.0
    mid_x = (vertices[:, 0].max() + vertices[:, 0].min()) * 0.5
    mid_y = (vertices[:, 1].max() + vertices[:, 1].min()) * 0.5
    mid_z = (vertices[:, 2].max() + vertices[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def visualize_point_cloud_matplotlib(points: np.ndarray, colors: Optional[np.ndarray] = None,
                                     title: str = "Point Cloud", save_path: Optional[Path] = None):
    """Visualize point cloud using matplotlib."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is not None:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c=colors, s=1, alpha=0.6)
    else:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                   c='blue', s=1, alpha=0.6)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


def densify_point_cloud(points: np.ndarray, colors: Optional[np.ndarray] = None, 
                        factor: int = 2, method: str = 'mesh_interpolation'):
    """
    Densify a point cloud by adding interpolated points.
    
    Args:
        points: Input point cloud (N, 3)
        colors: Optional colors (N, 3)
        factor: Densification factor (2 = double the points)
        method: 'mesh_interpolation' or 'knn_interpolation'
    
    Returns:
        Dense points and colors
    """
    if factor <= 1:
        return points, colors
    
    if method == 'mesh_interpolation' and TRIMESH_AVAILABLE:
        # Create mesh from points and interpolate on surface
        try:
            # Create a simple mesh from points using Delaunay triangulation
            from scipy.spatial import Delaunay
            # Project to 2D for triangulation (use XY plane)
            tri = Delaunay(points[:, :2])
            
            # Get triangle centers and interpolate
            new_points = []
            new_colors = []
            
            for simplex in tri.simplices:
                triangle_points = points[simplex]
                triangle_center = triangle_points.mean(axis=0)
                
                # Add points along edges
                for i in range(factor - 1):
                    alpha = (i + 1) / factor
                    # Interpolate along each edge
                    for j in range(3):
                        p1 = triangle_points[j]
                        p2 = triangle_points[(j + 1) % 3]
                        new_p = p1 * (1 - alpha) + p2 * alpha
                        new_points.append(new_p)
                        
                        if colors is not None:
                            c1 = colors[simplex[j]]
                            c2 = colors[simplex[(j + 1) % 3]]
                            new_c = c1 * (1 - alpha) + c2 * alpha
                            new_colors.append(new_c)
            
            if new_points:
                new_points = np.array(new_points)
                dense_points = np.vstack([points, new_points])
                
                if colors is not None and new_colors:
                    new_colors = np.array(new_colors)
                    dense_colors = np.vstack([colors, new_colors])
                else:
                    dense_colors = colors
                
                return dense_points, dense_colors
        except Exception as e:
            print(f"  Warning: Mesh interpolation failed: {e}, using simple method")
    
    # Fallback: Simple KNN interpolation
    if OPEN3D_AVAILABLE:
        try:
            import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            if colors is not None:
                if colors.max() <= 1.0:
                    colors_normalized = colors
                else:
                    colors_normalized = colors / 255.0
                pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
            
            # Use Poisson disk sampling or uniform downsampling in reverse
            # For densification, we'll add noise-based points
            num_new = len(points) * (factor - 1)
            # Get bounding box
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = bbox.get_extent()
            
            # Generate random points near existing points
            new_points = []
            new_colors = []
            for _ in range(num_new):
                # Pick random existing point
                idx = np.random.randint(len(points))
                base_point = points[idx]
                # Add small random offset
                offset = np.random.normal(0, extent * 0.01, 3)
                new_points.append(base_point + offset)
                
                if colors is not None:
                    new_colors.append(colors[idx])
            
            if new_points:
                new_points = np.array(new_points)
                dense_points = np.vstack([points, new_points])
                
                if colors is not None and new_colors:
                    new_colors = np.array(new_colors)
                    dense_colors = np.vstack([colors, new_colors])
                else:
                    dense_colors = colors
                
                return dense_points, dense_colors
        except Exception as e:
            print(f"  Warning: Open3D densification failed: {e}")
    
    # Final fallback: Just duplicate points with small noise
    num_new = len(points) * (factor - 1)
    noise_scale = np.std(points, axis=0).mean() * 0.01
    new_points = points[np.random.choice(len(points), num_new, replace=True)]
    new_points = new_points + np.random.normal(0, noise_scale, new_points.shape)
    
    dense_points = np.vstack([points, new_points])
    
    if colors is not None:
        new_colors = colors[np.random.choice(len(colors), num_new, replace=True)]
        dense_colors = np.vstack([colors, new_colors])
    else:
        dense_colors = None
    
    return dense_points, dense_colors


def visualize_with_rerun(vertices: Optional[np.ndarray] = None,
                         faces: Optional[np.ndarray] = None,
                         points: Optional[np.ndarray] = None,
                         colors: Optional[np.ndarray] = None,
                         cameras: Optional[list] = None,
                         vertex_colors: Optional[np.ndarray] = None,
                         show_mesh_as_points: bool = False,
                         max_points: int = 0,  # 0 = no limit, show all points
                         densify_factor: int = 1,
                         entity_path: str = "scene"):
    """Visualize using Rerun (excellent for 3D point clouds)."""
    if not RERUN_AVAILABLE:
        print("Rerun not available. Install with: pip install rerun-sdk")
        return False
    
    # Initialize Rerun
    rr.init("Point Cloud Visualization", spawn=True)
    
    # Log COLMAP points (separate from mesh points)
    if points is not None:
        print(f"Logging {len(points)} COLMAP reconstruction points to Rerun...")
        
        # Prepare colors
        if colors is not None:
            colors_array = np.array(colors)
            # Ensure colors are in [0, 255] range for Rerun
            if colors_array.max() <= 1.0:
                colors_array = (colors_array * 255).astype(np.uint8)
            else:
                colors_array = colors_array.astype(np.uint8)
            # Ensure shape is (N, 3) or (N, 4)
            if colors_array.ndim == 1:
                colors_array = np.tile(colors_array, (len(points), 1))
            elif colors_array.shape[0] != len(points):
                colors_array = np.tile(colors_array, (len(points), 1))
            # Convert to list of RGB tuples for Rerun
            colors_list = [tuple(int(c[i]) for i in range(3)) for c in colors_array]
        else:
            # Default yellow color for COLMAP points to distinguish from mesh
            colors_list = [(255, 255, 0)] * len(points)  # Yellow
        
        # Log COLMAP points to a separate path
        rr.log(
            f"{entity_path}/colmap_points",
            rr.Points3D(
                positions=points,
                colors=colors_list,
                radii=0.02  # Slightly larger to distinguish from mesh points
            )
        )
        print(f"✓ COLMAP points logged to '{entity_path}/colmap_points'")
    
    # Log mesh surface only if NOT showing as points (to avoid clumping)
    if vertices is not None and faces is not None and not show_mesh_as_points:
        print(f"Logging mesh surface with {len(vertices)} vertices, {len(faces)} faces...")
        
        # Rerun expects mesh as triangle list
        rr.log(
            f"{entity_path}/mesh",
            rr.Mesh3D(
                vertex_positions=vertices,
                triangle_indices=faces,
            )
        )
        print(f"✓ Mesh surface logged to '{entity_path}/mesh'")
    
    # Log mesh vertices as point cloud (if requested or if vertex colors available)
    if vertices is not None and show_mesh_as_points:
        print(f"Logging mesh vertices as point cloud ({len(vertices)} points)...")
        
        # Optionally downsample if too many points (for better visualization)
        # Only downsample if max_points is set and we exceed it
        if max_points > 0 and len(vertices) > max_points:
            print(f"  Downsampling from {len(vertices)} to {max_points} points for better visualization...")
            indices = np.random.choice(len(vertices), max_points, replace=False)
            vertices_display = vertices[indices]
            if vertex_colors is not None:
                vertex_colors_display = vertex_colors[indices]
            else:
                vertex_colors_display = None
        else:
            vertices_display = vertices
            vertex_colors_display = vertex_colors
        
        # Densify if requested
        if densify_factor > 1:
            print(f"  Densifying point cloud by factor of {densify_factor}...")
            vertices_display, vertex_colors_display = densify_point_cloud(
                vertices_display, vertex_colors_display, 
                factor=densify_factor
            )
            print(f"  Dense point cloud: {len(vertices_display)} points")
        
        if vertex_colors_display is not None:
            # Use vertex colors if available
            colors_list = [tuple(int(c[i]) for i in range(3)) for c in vertex_colors_display]
        else:
            # Color by position (height-based coloring)
            z_min, z_max = vertices_display[:, 2].min(), vertices_display[:, 2].max()
            z_normalized = (vertices_display[:, 2] - z_min) / (z_max - z_min + 1e-8)
            # Create colormap: blue (low) -> green -> red (high)
            colors_list = []
            for z_norm in z_normalized:
                if z_norm < 0.5:
                    # Blue to green
                    r, g, b = 0, int(z_norm * 2 * 255), int((1 - z_norm * 2) * 255)
                else:
                    # Green to red
                    r, g, b = int((z_norm - 0.5) * 2 * 255), int((1 - (z_norm - 0.5) * 2) * 255), 0
                colors_list.append((r, g, b))
        
        rr.log(
            f"{entity_path}/points",
            rr.Points3D(
                positions=vertices_display,
                colors=colors_list,
                radii=0.01
            )
        )
        print(f"✓ Mesh vertices logged as point cloud to '{entity_path}/points' ({len(vertices_display)} points)")
    
    # Log camera poses if available
    if cameras is not None:
        print(f"Logging {len(cameras)} camera poses...")
        for i, cam_pose in enumerate(cameras):
            # Convert to Rerun transform format
            translation = cam_pose[:3, 3]
            rotation_matrix = cam_pose[:3, :3]
            # Rerun uses quaternion or rotation matrix
            rr.log(
                f"{entity_path}/cameras/camera_{i}",
                rr.Transform3D(
                    translation=translation,
                    mat3x3=rotation_matrix
                )
            )
        print(f"✓ Camera poses logged to '{entity_path}/cameras'")
    
    print("\n=== Rerun Viewer ===")
    print("The Rerun viewer window should open automatically.")
    print("You can interact with the 3D scene in the viewer.")
    print("Close the viewer window or press Ctrl+C to exit.")
    print("===================\n")
    
    # Keep the script running so the viewer stays open
    try:
        import time
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nClosing Rerun viewer...")
    
    return True


def visualize_with_open3d(vertices: Optional[np.ndarray] = None,
                          faces: Optional[np.ndarray] = None,
                          points: Optional[np.ndarray] = None,
                          colors: Optional[np.ndarray] = None,
                          cameras: Optional[list] = None,
                          window_name: str = "3D Visualization"):
    """Visualize using Open3D (interactive)."""
    if not OPEN3D_AVAILABLE:
        print("Open3D not available. Falling back to matplotlib.")
        return False
    
    geometries = []
    
    # Add mesh
    if vertices is not None and faces is not None:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])
        geometries.append(mesh)
    
    # Add point cloud
    if points is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            # Ensure colors are in [0, 1] range
            if colors.max() > 1.0:
                colors = colors / 255.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd.paint_uniform_color([0.0, 0.0, 1.0])
        geometries.append(pcd)
    
    # Add camera frustums
    if cameras is not None:
        for cam_pose in cameras:
            # Create camera frustum
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                view_width_px=640, view_height_px=480, 
                intrinsic=o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240),
                extrinsic=np.linalg.inv(cam_pose)
            )
            geometries.append(frustum)
    
    if geometries:
        # Create visualizer with better controls
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=window_name, width=1920, height=1080)
        
        for geom in geometries:
            vis.add_geometry(geom)
        
        # Set up render options
        render_option = vis.get_render_option()
        render_option.point_size = 2.0
        render_option.background_color = np.asarray([0.1, 0.1, 0.1])
        
        # Set up view control
        view_control = vis.get_view_control()
        view_control.set_zoom(0.7)
        
        print("\n=== Open3D Interactive Viewer ===")
        print("Controls:")
        print("  - Mouse drag: Rotate")
        print("  - Shift + Mouse drag: Pan")
        print("  - Mouse wheel: Zoom")
        print("  - R: Reset view")
        print("  - Q or close window: Quit")
        print("===============================\n")
        
        vis.run()
        vis.destroy_window()
        return True
    return False


def visualize_shaper_pickle(pickle_path: Path, use_open3d: bool = True, 
                            use_rerun: bool = False, save_path: Optional[Path] = None):
    """Visualize ShapeR pickle file."""
    print(f"Loading ShapeR pickle: {pickle_path}")
    data = load_shaper_pickle(pickle_path)
    
    # Extract point cloud
    if 'points_model' in data:
        points = data['points_model']
        if TORCH_AVAILABLE and isinstance(points, torch.Tensor):
            points = points.numpy()
        elif not isinstance(points, np.ndarray):
            points = np.array(points)
        print(f"Point cloud shape: {points.shape}")
        
        # Use Rerun if requested
        if use_rerun and RERUN_AVAILABLE and save_path is None:
            visualize_with_rerun(points=points, entity_path=f"shaper/{pickle_path.stem}")
        # Use Open3D if available and not saving to file
        elif save_path is None and (use_open3d or OPEN3D_AVAILABLE):
            if OPEN3D_AVAILABLE:
                visualize_with_open3d(points=points, window_name=f"ShapeR Point Cloud: {pickle_path.name}")
            else:
                print("Open3D not available, falling back to matplotlib")
                visualize_point_cloud_matplotlib(
                    points, 
                    title=f"ShapeR Point Cloud: {pickle_path.name}",
                    save_path=save_path
                )
        else:
            visualize_point_cloud_matplotlib(
                points, 
                title=f"ShapeR Point Cloud: {pickle_path.name}",
                save_path=save_path
            )
    
    # Extract mesh if available
    if 'mesh_vertices' in data and 'mesh_faces' in data:
        vertices = data['mesh_vertices']
        faces = data['mesh_faces']
        if TORCH_AVAILABLE and isinstance(vertices, torch.Tensor):
            vertices = vertices.numpy()
        elif not isinstance(vertices, np.ndarray):
            vertices = np.array(vertices)
        if TORCH_AVAILABLE and isinstance(faces, torch.Tensor):
            faces = faces.numpy()
        elif not isinstance(faces, np.ndarray):
            faces = np.array(faces)
        print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        if save_path is None and (use_open3d or OPEN3D_AVAILABLE):
            if OPEN3D_AVAILABLE:
                visualize_with_open3d(vertices=vertices, faces=faces, 
                                    window_name=f"ShapeR Mesh: {pickle_path.name}")
            else:
                visualize_mesh_matplotlib(
                    vertices, faces,
                    title=f"ShapeR Mesh: {pickle_path.name}",
                    save_path=save_path
                )
        else:
            visualize_mesh_matplotlib(
                vertices, faces,
                title=f"ShapeR Mesh: {pickle_path.name}",
                save_path=save_path
            )


def align_coordinate_systems(points: np.ndarray, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align points and mesh vertices to the same coordinate system.
    Uses centroid alignment and optional scale normalization.
    """
    # Compute centroids
    points_centroid = points.mean(axis=0)
    vertices_centroid = vertices.mean(axis=0)
    
    # Center both
    points_centered = points - points_centroid
    vertices_centered = vertices - vertices_centroid
    
    # Compute scales
    points_scale = np.abs(points_centered).max()
    vertices_scale = np.abs(vertices_centered).max()
    
    # Normalize to same scale (use the larger scale as reference)
    if points_scale > 0 and vertices_scale > 0:
        scale_ratio = vertices_scale / points_scale
        # Option 1: Scale points to match mesh scale
        points_aligned = points_centered * scale_ratio + vertices_centroid
        vertices_aligned = vertices_centered + vertices_centroid
    else:
        # If scaling fails, just center them together
        points_aligned = points_centered + vertices_centroid
        vertices_aligned = vertices_centered + vertices_centroid
    
    return points_aligned, vertices_aligned


def visualize_scannet_scene(data_dir: Path, camera_source: str = 'iphone',
                           use_open3d: bool = True, use_rerun: bool = False,
                           save_path: Optional[Path] = None, align_coords: bool = True,
                           show_mesh_as_points: bool = True, show_colmap_points: bool = False,
                           max_points: int = 0, densify_factor: int = 1):  # 0 = no limit, show all points
    """Visualize ScanNet scene.
    
    Args:
        show_mesh_as_points: Show mesh vertices as point cloud (default: True)
        show_colmap_points: Show COLMAP sparse reconstruction points (default: False)
    """
    print(f"Loading ScanNet scene from: {data_dir}")
    
    geometries_to_visualize = []
    
    # Load mesh
    mesh_path = data_dir / 'scans' / 'mesh_aligned_0.05.ply'
    vertices = None
    faces = None
    vertex_colors = None
    if mesh_path.exists():
        print(f"Loading mesh: {mesh_path}")
        vertices, faces, vertex_colors = load_mesh_ply(mesh_path)
        print(f"Mesh: {len(vertices)} vertices, {len(faces)} faces")
        if vertex_colors is not None:
            print(f"Mesh has vertex colors")
        print(f"Mesh bounds: min={vertices.min(axis=0)}, max={vertices.max(axis=0)}")
        geometries_to_visualize.append(('mesh', vertices, faces, vertex_colors))
    else:
        print(f"Mesh not found at {mesh_path}")
    
    # Load COLMAP data
    cameras, images, points3d = load_colmap_data(data_dir, camera_source)
    print(f"Loaded {len(cameras)} cameras, {len(images)} images, {len(points3d)} 3D points")
    
    points = None
    colors = None
    if points3d and show_colmap_points:
        points = np.array([p['position'] for p in points3d.values()])
        colors = np.array([p['color'] for p in points3d.values()])
        print(f"COLMAP points: {len(points)} sparse reconstruction points")
        print(f"Points bounds: min={points.min(axis=0)}, max={points.max(axis=0)}")
        
        # Align coordinate systems if both mesh and points exist
        if vertices is not None and align_coords:
            print("\nAligning coordinate systems...")
            points_aligned, vertices_aligned = align_coordinate_systems(points, vertices)
            print(f"Aligned points bounds: min={points_aligned.min(axis=0)}, max={points_aligned.max(axis=0)}")
            print(f"Aligned mesh bounds: min={vertices_aligned.min(axis=0)}, max={vertices_aligned.max(axis=0)}")
            points = points_aligned
            vertices = vertices_aligned
            # Update the mesh in geometries_to_visualize
            if geometries_to_visualize and geometries_to_visualize[0][0] == 'mesh':
                geometries_to_visualize[0] = ('mesh', vertices, faces, vertex_colors)
        
        geometries_to_visualize.append(('colmap_points', points, colors))
    elif points3d:
        print(f"COLMAP points available ({len(points3d)} points) but not showing (use --show_colmap to display)")
    
    # Use Rerun if requested
    if use_rerun and RERUN_AVAILABLE and save_path is None:
        vertices_combined = None
        faces_combined = None
        vertex_colors_combined = None
        points_combined = None
        colors_combined = None
        
        for item in geometries_to_visualize:
            geom_type = item[0]
            if geom_type == 'mesh':
                vertices_combined, faces_combined, vertex_colors_combined = item[1], item[2], item[3] if len(item) > 3 else None
            elif geom_type == 'colmap_points':
                points_combined, colors_combined = item[1], item[2]
        
        # Get camera poses from images
        cameras_list = []
        if images:
            cameras_list = [img['pose'] for img in images.values()]
        
        visualize_with_rerun(
            vertices=vertices_combined,
            faces=faces_combined,
            points=points_combined,
            colors=colors_combined,
            cameras=cameras_list if cameras_list else None,
            vertex_colors=vertex_colors_combined,
            show_mesh_as_points=show_mesh_as_points,
            max_points=max_points,
            densify_factor=densify_factor,
            entity_path=f"scannet/{data_dir.name}"
        )
    # Visualize all geometries together if using Open3D
    elif save_path is None and (use_open3d or OPEN3D_AVAILABLE) and OPEN3D_AVAILABLE:
        # Combine all geometries for Open3D
        vertices_combined = None
        faces_combined = None
        points_combined = None
        colors_combined = None
        
        for geom_type, *geom_data in geometries_to_visualize:
            if geom_type == 'mesh':
                vertices_combined, faces_combined = geom_data
            elif geom_type == 'points':
                points_combined, colors_combined = geom_data
        
        visualize_with_open3d(
            vertices=vertices_combined,
            faces=faces_combined,
            points=points_combined,
            colors=colors_combined,
            window_name=f"ScanNet Scene: {data_dir.name}"
        )
    else:
        # Fall back to matplotlib for each geometry
        for geom_type, *geom_data in geometries_to_visualize:
            if geom_type == 'mesh':
                vertices, faces = geom_data
                visualize_mesh_matplotlib(
                    vertices, faces,
                    title=f"ScanNet Mesh: {data_dir.name}",
                    save_path=save_path
                )
            elif geom_type == 'points':
                points, colors = geom_data
                visualize_point_cloud_matplotlib(
                    points, colors,
                    title=f"COLMAP Points: {data_dir.name}",
                    save_path=save_path
                )


def main():
    parser = argparse.ArgumentParser(description='Visualize ScanNet scene data')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to data (ScanNet directory, pickle file, or mesh file)')
    parser.add_argument('--camera_source', type=str, default='iphone',
                       choices=['iphone', 'dslr'],
                       help='Camera source for ScanNet (default: iphone)')
    parser.add_argument('--use_open3d', action='store_true',
                       help='Use Open3D for interactive visualization (default: True if open3d available)')
    parser.add_argument('--use_rerun', action='store_true',
                       help='Use Rerun for interactive 3D visualization (excellent for point clouds)')
    parser.add_argument('--use_matplotlib', action='store_true',
                       help='Force use matplotlib instead of Open3D/Rerun')
    parser.add_argument('--save', type=str, default=None,
                       help='Save visualization to file (instead of displaying)')
    parser.add_argument('--type', type=str, default='auto',
                       choices=['auto', 'scannet', 'shaper', 'mesh', 'points'],
                       help='Data type (auto-detect if not specified)')
    parser.add_argument('--no_align', action='store_true',
                       help='Do not align coordinate systems (useful if points and mesh are already aligned)')
    parser.add_argument('--show_colmap', action='store_true',
                       help='Show COLMAP sparse reconstruction points (default: only show mesh point cloud)')
    parser.add_argument('--no_mesh_points', action='store_true',
                       help='Show mesh surface instead of point cloud (default: show mesh as point cloud)')
    parser.add_argument('--max_points', type=int, default=0,
                       help='Maximum number of points to display (downsamples if exceeded, default: 0=no limit, show all points)')
    parser.add_argument('--densify', type=int, default=1,
                       help='Densify point cloud by this factor (e.g., 2 = double the points, default: 1 = no densification)')
    
    args = parser.parse_args()
    data_path = Path(args.data_path)
    
    if not data_path.exists():
        print(f"Error: Path does not exist: {data_path}")
        return
    
    save_path = Path(args.save) if args.save else None
    
    # Auto-detect type
    if args.type == 'auto':
        if data_path.is_file():
            if data_path.suffix == '.pkl':
                args.type = 'shaper'
            elif data_path.suffix == '.ply':
                args.type = 'mesh'
            else:
                args.type = 'scannet'
        else:
            # Check if it's a ScanNet directory
            if (data_path / 'scans' / 'mesh_aligned_0.05.ply').exists():
                args.type = 'scannet'
            else:
                args.type = 'scannet'  # Default
    
    # Determine visualization backend
    use_rerun = args.use_rerun and RERUN_AVAILABLE and save_path is None
    use_open3d = (args.use_open3d or (not args.use_matplotlib and not use_rerun and OPEN3D_AVAILABLE)) and not args.use_matplotlib and not use_rerun
    
    if use_rerun:
        print("Using Rerun for interactive 3D visualization")
    elif use_open3d and save_path is None:
        print("Using Open3D for interactive 3D visualization")
    elif save_path:
        print(f"Saving visualization to {save_path} (using matplotlib)")
    else:
        print("Using matplotlib for visualization")
    
    # Visualize based on type
    if args.type == 'shaper':
        visualize_shaper_pickle(data_path, use_open3d, use_rerun, save_path)
    elif args.type == 'mesh':
        vertices, faces = load_mesh_ply(data_path)
        if use_rerun:
            visualize_with_rerun(vertices=vertices, faces=faces, 
                               entity_path=f"mesh/{data_path.stem}")
        elif use_open3d and save_path is None:
            visualize_with_open3d(vertices=vertices, faces=faces, 
                                window_name=f"Mesh: {data_path.name}")
        else:
            visualize_mesh_matplotlib(vertices, faces, 
                                   title=f"Mesh: {data_path.name}",
                                   save_path=save_path)
    elif args.type == 'scannet':
        visualize_scannet_scene(data_path, args.camera_source, 
                              use_open3d, use_rerun, save_path, 
                              align_coords=not args.no_align,
                              show_mesh_as_points=not args.no_mesh_points,
                              show_colmap_points=args.show_colmap,
                              max_points=args.max_points,
                              densify_factor=args.densify)
    else:
        print(f"Unknown visualization type: {args.type}")


if __name__ == '__main__':
    main()
