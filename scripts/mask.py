#!/usr/bin/env python3
"""
Generate Instance Segmentation Masks from Annotated PLY

Takes annotated PLY mesh and segments_anno.json to create 2D instance segmentation masks
for each object in each image by projecting 3D points to 2D.

Usage:
    python scripts/mask.py --data_dir data/00777c41d4 --output_dir output_masks --camera_source iphone
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import trimesh
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.distance import cdist
from scipy.ndimage import binary_fill_holes


def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


# REMOVED: align_coordinate_systems function
# ScanNet++ COLMAP is already aligned with the mesh per documentation:
# "contains the colmap camera model that has been aligned with the 3D scans"
# So alignment would break the existing alignment!


def load_colmap_data(colmap_dir: Path) -> Tuple[Dict, Dict, Dict]:
    """Load COLMAP cameras, images, and points."""
    def load_cameras(cameras_path: Path) -> Dict:
        cameras = {}
        with open(cameras_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                cam_id = int(parts[0])
                model = parts[1]
                width = int(parts[2])
                height = int(parts[3])
                params = [float(p) for p in parts[4:]]
                cameras[cam_id] = {
                    'model': model,
                    'width': width,
                    'height': height,
                    'params': params
                }
        return cameras
    
    def load_images(images_path: Path) -> Dict:
        images = {}
        with open(images_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith('#'):
                    i += 1
                    continue
                parts = line.split()
                if len(parts) < 10:
                    i += 1
                    continue
                img_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                cam_id = int(parts[8])
                name = parts[9]
                
                R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                # COLMAP format: According to COLMAP docs, quaternion and translation
                # represent the camera pose in world coordinates (camera-to-world).
                # However, for projection we need world-to-camera.
                # But based on scannet_to_shaper.py, it seems COLMAP format might already be world-to-camera.
                # Let's try both and see which works - for now, use direct (no inversion)
                # as that matches scannet_to_shaper.py behavior
                T_world_cam = np.eye(4)
                T_world_cam[:3, :3] = R
                T_world_cam[:3, 3] = [tx, ty, tz]
                
                images[img_id] = {
                    'name': name,
                    'cam_id': cam_id,
                    'T_world_cam': T_world_cam,  # World to camera (for projection)
                }
                i += 1
        return images
    
    def load_points(points_path: Path) -> Dict:
        points = {}
        with open(points_path, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                point_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points[point_id] = np.array([x, y, z])
        return points
    
    cameras = load_cameras(colmap_dir / 'cameras.txt')
    images = load_images(colmap_dir / 'images.txt')
    points = load_points(colmap_dir / 'points3D.txt')
    
    return cameras, images, points


def get_object_points_from_segments(segment_ids: List[int], mesh_vertices: np.ndarray,
                                    seg_indices: List[int], colmap_points: Dict = None,
                                    distance_threshold: float = 0.05) -> np.ndarray:
    """
    Extract 3D points belonging to an object from segment IDs.
    
    Args:
        segment_ids: List of segment IDs for this object (from segments_anno.json)
        mesh_vertices: Mesh vertex positions
        seg_indices: Per-vertex segment IDs from segments.json (length = num_vertices)
        colmap_points: Optional COLMAP points for denser coverage
        distance_threshold: Distance threshold for matching COLMAP points
    
    Returns:
        Object 3D points (mesh vertices + matched COLMAP points)
    """
    # Map segment IDs to vertex indices using segments.json
    object_segment_ids = set(segment_ids)
    vertex_mask = [seg_indices[i] in object_segment_ids for i in range(len(seg_indices))]
    object_vertices = mesh_vertices[vertex_mask]
    
    if len(object_vertices) == 0:
        return np.array([]).reshape(0, 3)
    
    # Optionally match COLMAP points for denser coverage
    if colmap_points and len(colmap_points) > 0:
        colmap_points_array = np.array(list(colmap_points.values()))
        
        if len(colmap_points_array) > 0:
            mesh_tree = cKDTree(object_vertices)
            distances, _ = mesh_tree.query(colmap_points_array, k=1, distance_upper_bound=distance_threshold)
            valid_mask = distances < distance_threshold
            matched_colmap_points = colmap_points_array[valid_mask]
            
            if len(matched_colmap_points) > 0:
                # Combine COLMAP points and mesh vertices
                all_points = np.vstack([matched_colmap_points, object_vertices])
                # Remove duplicates
                _, unique_indices = np.unique(
                    np.round(all_points / 0.01).astype(int), 
                    axis=0, 
                    return_index=True
                )
                return all_points[unique_indices]
    
    return object_vertices


def project_points_to_image(points_3d: np.ndarray, T_world_cam: np.ndarray,
                           camera_params: np.ndarray, width: int, height: int,
                           camera_model: str = 'OPENCV',
                           depth_map: np.ndarray = None,
                           occlusion_threshold: float = 0.1,
                           debug: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D world points to 2D image coordinates with aggressive filtering to prevent ghosting.
    
    Args:
        points_3d: (N, 3) points in world coordinates
        T_world_cam: (4, 4) transformation from world to camera (COLMAP format)
        camera_params: Camera intrinsics [fx, fy, cx, cy, k1, k2, p1, p2, ...]
        width: Image width
        height: Image height
        camera_model: Camera model type ('OPENCV' or 'OPENCV_FISHEYE')
        depth_map: Optional (H, W) depth map for occlusion checking. Points occluded by
                   closer points in depth_map will be filtered out.
        occlusion_threshold: Depth difference threshold (in meters) for occlusion checking.
                            Points deeper than existing_depth + threshold are considered occluded.
    
    Returns:
        (points_2d, valid_mask) where valid_mask indicates points within image bounds and not occluded
    """
    # Transform points from world to camera frame
    points_cam = (T_world_cam[:3, :3] @ points_3d.T).T + T_world_cam[:3, 3]
    
    # FOV cutoff: Filter points with extreme viewing angles (near 90°)
    # Points at extreme angles often cause streaks
    # Compute viewing angle: angle between camera Z-axis and point direction
    point_norms_xy = np.linalg.norm(points_cam[:, :2], axis=1)
    point_depths = points_cam[:, 2]
    
    # Viewing angle from camera Z-axis (0° = straight ahead, 90° = edge of FOV)
    # tan(angle) = sqrt(x^2 + y^2) / z
    viewing_angles = np.arctan2(point_norms_xy, np.abs(point_depths) + 1e-6)
    max_viewing_angle = np.deg2rad(75)  # 75 degrees max FOV - eliminates near-90° points
    valid_fov = viewing_angles < max_viewing_angle
    
    # Aggressive depth filtering to prevent ghosting
    min_depth = 0.2  # 20cm minimum depth (more aggressive)
    max_depth = 50.0  # 50m maximum depth (reasonable for indoor scenes)
    valid_depth = (points_cam[:, 2] > min_depth) & (points_cam[:, 2] < max_depth) & valid_fov
    
    if not np.any(valid_depth):
        return np.array([]).reshape(0, 2), np.array([], dtype=bool)
    
    points_cam_valid = points_cam[valid_depth]
    
    # Extract camera parameters
    fx, fy = camera_params[0], camera_params[1]
    cx, cy = camera_params[2], camera_params[3]
    
    # Project based on camera model
    if camera_model == 'OPENCV_FISHEYE':
        dist_coeffs = camera_params[4:8] if len(camera_params) >= 8 else np.zeros(4)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        try:
            points_2d, _ = cv2.fisheye.projectPoints(
                points_cam_valid.reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                dist_coeffs
            )
        except:
            points_2d, _ = cv2.projectPoints(
                points_cam_valid.reshape(-1, 1, 3),
                np.zeros(3),
                np.zeros(3),
                camera_matrix,
                dist_coeffs
            )
    else:
        dist_coeffs = camera_params[4:8] if len(camera_params) >= 8 else np.zeros(4)
        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        points_2d, _ = cv2.projectPoints(
            points_cam_valid.reshape(-1, 1, 3),
            np.zeros(3),
            np.zeros(3),
            camera_matrix,
            dist_coeffs
        )
    
    points_2d = points_2d.reshape(-1, 2)
    
    # Aggressive filtering to prevent ghosting/smearing:
    # 1. Filter out points projecting too close to center (invalid/at infinity)
    center_threshold = 20.0  # pixels - even more aggressive for walls/floors
    dist_from_center = np.sqrt((points_2d[:, 0] - cx)**2 + (points_2d[:, 1] - cy)**2)
    not_at_center = dist_from_center > center_threshold
    
    # 2. Filter NaN/Inf values
    valid_coords = np.isfinite(points_2d).all(axis=1)
    
    # 3. Points within image bounds with larger margin (avoid edge artifacts)
    margin = 10  # pixels margin from edges - increased for walls/floors
    in_bounds = (points_2d[:, 0] >= margin) & (points_2d[:, 0] < width - margin) & \
                (points_2d[:, 1] >= margin) & (points_2d[:, 1] < height - margin)
    
    # 4. Filter points that are too far from the principal axis (likely invalid for fisheye)
    if camera_model == 'OPENCV_FISHEYE':
        max_dist = min(width, height) * 0.55  # 55% of image dimension - more conservative
        reasonable_fov = dist_from_center < max_dist
    else:
        reasonable_fov = np.ones(len(points_2d), dtype=bool)
    
    # 5. Additional filtering: Remove points with extreme depth ratios (likely occluded/invalid)
    # Points that are much closer or farther than the median depth are suspicious
    if len(points_cam_valid) > 10:
        median_depth = np.median(points_cam_valid[:, 2])
        depth_ratio = points_cam_valid[:, 2] / (median_depth + 1e-6)
        # Keep points within 0.3x to 3x of median depth
        reasonable_depth = (depth_ratio > 0.3) & (depth_ratio < 3.0)
    else:
        reasonable_depth = np.ones(len(points_cam_valid), dtype=bool)
    
    # 6. Filter points with extreme angles (likely self-occluded or invalid)
    # Points with very shallow viewing angles are often problematic
    viewing_angle = np.abs(points_cam_valid[:, 2]) / (np.linalg.norm(points_cam_valid[:, :2], axis=1) + 1e-6)
    reasonable_angle = viewing_angle > 0.1  # At least 10 degrees from perpendicular
    
    # 7. Occlusion check: Filter points that are occluded by other objects
    # A point is occluded if there's a closer point at the same pixel location
    valid_occlusion = np.ones(len(points_2d), dtype=bool)
    num_occluded = 0
    if depth_map is not None:
        x_int = np.clip(points_2d[:, 0].astype(int), 0, width - 1)
        y_int = np.clip(points_2d[:, 1].astype(int), 0, height - 1)
        
        # Check if projected depth is significantly closer than existing depth map
        # occlusion_threshold allows some tolerance (e.g., 10cm) to handle noise
        for i in range(len(points_2d)):
            existing_depth = depth_map[y_int[i], x_int[i]]
            point_depth = points_cam_valid[i, 2]
            
            # Point is occluded if existing depth is closer (smaller) by more than threshold
            if existing_depth < np.inf and point_depth > existing_depth + occlusion_threshold:
                valid_occlusion[i] = False
                num_occluded += 1
        
        if debug:
            print(f"    Occlusion check: {num_occluded}/{len(points_2d)} points occluded")
            if num_occluded > 0:
                # Show some examples
                occluded_indices = np.where(~valid_occlusion)[0][:5]
                for idx in occluded_indices:
                    print(f"      Point {idx}: depth={points_cam_valid[idx, 2]:.2f}m, existing={depth_map[y_int[idx], x_int[idx]]:.2f}m, diff={points_cam_valid[idx, 2] - depth_map[y_int[idx], x_int[idx]]:.2f}m")
    
    # Combine all validity checks
    valid_projection = in_bounds & not_at_center & valid_coords & reasonable_fov & reasonable_depth & reasonable_angle & valid_occlusion
    
    # Create full output arrays
    full_points_2d = np.zeros((len(points_3d), 2), dtype=np.float32)
    full_valid = np.zeros(len(points_3d), dtype=bool)
    full_points_2d[valid_depth] = points_2d
    full_valid[valid_depth] = valid_projection
    
    return full_points_2d, full_valid


def points_to_mask(points_2d: np.ndarray, width: int, height: int, 
                   fill_holes: bool = True, dilate: bool = True) -> np.ndarray:
    """
    Convert 2D point projections to a binary mask by drawing circles around each point.
    This prevents streaks by not connecting scattered points with convex hull.
    """
    if len(points_2d) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Clip points to image bounds
    x = np.clip(points_2d[:, 0], 0, width - 1)
    y = np.clip(points_2d[:, 1], 0, height - 1)
    
    # Remove duplicate points to prevent over-drawing
    unique_points = np.unique(np.column_stack([x, y]), axis=0)
    if len(unique_points) == 0:
        return np.zeros((height, width), dtype=np.uint8)
    
    # Create mask by drawing circles around each point
    mask = np.zeros((height, width), dtype=np.uint8)
    point_radius = max(1, int(min(width, height) / 800))  # Small radius to prevent smearing
    
    for px, py in unique_points.astype(int):
        cv2.circle(mask, (int(px), int(py)), point_radius, 255, -1)
    
    # Optional: fill small holes (only very small ones, not large gaps)
    if fill_holes:
        # Use morphological operations to fill only small holes
        kernel_small = np.ones((3, 3), np.uint8)
        # Close small gaps first
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
        # Then fill remaining small holes using distance transform
        # Only fill areas close to original points
        dist_transform = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
        close_to_points = dist_transform < 5  # Only fill within 5 pixels of original points
        mask_filled = binary_fill_holes(mask > 0).astype(np.uint8) * 255
        mask = np.where(close_to_points, mask_filled, mask)
    
    # Optional: minimal dilation to smooth edges without smearing
    if dilate:
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    
    return mask


def generate_masks_for_image(img_id: int, img_info: Dict, objects_data: List[Dict],
                             mesh_vertices: np.ndarray, seg_indices: List[int],
                             cameras: Dict, colmap_points: Dict, output_dir: Path,
                             rgb_dir: Path = None,
                             depth_dir: Path = None,
                             create_binary: bool = True, create_instance: bool = True,
                             fill_holes: bool = True, dilate: bool = True,
                             min_points: int = 5) -> Dict:
    """Generate masks for all objects in a single image."""
    img_name = img_info['name']
    cam_id = img_info['cam_id']
    
    if cam_id not in cameras:
        return None
    
    camera_info = cameras[cam_id]
    T_world_cam = img_info['T_world_cam']  # World to camera transform
    
    width = camera_info['width']
    height = camera_info['height']
    
    # Load original image if available
    original_image = None
    if rgb_dir is not None:
        img_path = rgb_dir / img_name
        if img_path.exists():
            original_image = cv2.imread(str(img_path))
            if original_image is not None:
                # Resize if needed
                if original_image.shape[1] != width or original_image.shape[0] != height:
                    original_image = cv2.resize(original_image, (width, height))
    
    # Load depth map from ScanNet depth images (much better than sparse COLMAP points!)
    # This provides dense, per-pixel depth for proper occlusion checking
    depth_map = np.full((height, width), np.inf, dtype=np.float32)
    
    if depth_dir is not None:
        # ScanNet depth images are named frame_XXXXXX.png matching RGB images
        # Extract frame number from RGB image name (e.g., "frame_000000.jpg" -> "frame_000000")
        img_name_base = img_name.replace('.jpg', '').replace('.JPG', '')
        depth_img_path = depth_dir / f"{img_name_base}.png"
        
        if depth_img_path.exists():
            # Load depth image (typically 16-bit PNG, values in millimeters)
            depth_img = cv2.imread(str(depth_img_path), cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
            
            if depth_img is not None:
                # Handle different depth image formats
                if len(depth_img.shape) == 3:
                    # If RGB, take first channel (shouldn't happen but handle it)
                    depth_img = depth_img[:, :, 0]
                
                # Convert to float32 depth in meters
                # ScanNet depth is typically in millimeters (0-65535 range for 16-bit)
                # But check actual range to determine units
                depth_max = depth_img.max()
                if depth_max > 10000:
                    # Likely in millimeters, convert to meters
                    depth_map = depth_img.astype(np.float32) / 1000.0
                elif depth_max > 100:
                    # Likely already in centimeters, convert to meters
                    depth_map = depth_img.astype(np.float32) / 100.0
                else:
                    # Likely already in meters
                    depth_map = depth_img.astype(np.float32)
                
                # Resize if needed to match camera dimensions
                if depth_map.shape[1] != width or depth_map.shape[0] != height:
                    depth_map = cv2.resize(depth_map, (width, height), interpolation=cv2.INTER_NEAREST)
                
                # Filter invalid depths (0 or very large values)
                depth_map[depth_map <= 0] = np.inf
                depth_map[depth_map > 50.0] = np.inf  # Max 50m for indoor scenes
        else:
            # Fallback: build sparse depth map from COLMAP points if depth image not found
            if colmap_points:
                # colmap_points values are already numpy arrays [x, y, z]
                all_colmap_points = np.array(list(colmap_points.values()))
                if len(all_colmap_points) > 0:
                    # Transform to camera frame
                    points_cam_all = (T_world_cam[:3, :3] @ all_colmap_points.T).T + T_world_cam[:3, 3]
                    # Filter by depth
                    valid_all = (points_cam_all[:, 2] > 0.2) & (points_cam_all[:, 2] < 50.0)
                    if np.any(valid_all):
                        points_cam_valid = points_cam_all[valid_all]
                        depths_valid = points_cam_valid[:, 2]
                        
                        # Project to image
                        fx, fy = camera_info['params'][0], camera_info['params'][1]
                        cx, cy = camera_info['params'][2], camera_info['params'][3]
                        camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                        
                        if camera_info.get('model', 'OPENCV') == 'OPENCV_FISHEYE':
                            dist_coeffs = np.array(camera_info['params'][4:8], dtype=np.float32) if len(camera_info['params']) >= 8 else np.zeros(4, dtype=np.float32)
                            try:
                                points_2d_all, _ = cv2.fisheye.projectPoints(
                                    points_cam_valid.reshape(-1, 1, 3),
                                    np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                                )
                            except:
                                points_2d_all, _ = cv2.projectPoints(
                                    points_cam_valid.reshape(-1, 1, 3),
                                    np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                                )
                        else:
                            dist_coeffs = np.array(camera_info['params'][4:8], dtype=np.float32) if len(camera_info['params']) >= 8 else np.zeros(4, dtype=np.float32)
                            points_2d_all, _ = cv2.projectPoints(
                                points_cam_valid.reshape(-1, 1, 3),
                                np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                            )
                        
                        points_2d_all = points_2d_all.reshape(-1, 2)
                        
                        # Update depth map: keep minimum depth at each pixel
                        x_int = np.clip(points_2d_all[:, 0].astype(int), 0, width - 1)
                        y_int = np.clip(points_2d_all[:, 1].astype(int), 0, height - 1)
                        
                        for i in range(len(x_int)):
                            if depths_valid[i] < depth_map[y_int[i], x_int[i]]:
                                depth_map[y_int[i], x_int[i]] = depths_valid[i]
    
    # Create instance mask (all objects in one image)
    instance_mask = np.zeros((height, width), dtype=np.uint16)
    image_objects = []
    
    for obj_data in objects_data:
        object_id = obj_data['id']
        label = obj_data.get('label', 'object')
        segments = obj_data['segments']
        
        # Debug mode for all objects (can be limited to specific objects/frames)
        debug_objects = None  # Set to [55, 56] to debug specific objects, None for all
        debug_frames = [0]  # Set to [0] to debug only frame 0, None for all frames
        debug_mode = (debug_objects is None or object_id in debug_objects) and \
                    (debug_frames is None or img_id in debug_frames)
        
        if debug_mode:
            print(f"\n=== DEBUG Object {object_id} ({label}) in frame {img_id} ===")
        
        # Skip structural elements (walls, floors, ceilings) - not needed for ShapeR
        label_lower = label.lower()
        structural_labels = ['wall', 'floor', 'ceiling', 'ground', 'door', 'window', 'doorframe']
        if any(struct_label in label_lower for struct_label in structural_labels):
            if debug_mode:
                print(f"  SKIPPED: Structural element")
            continue
        
        # Get object points using segment IDs mapped to vertices via seg_indices
        object_points_world = get_object_points_from_segments(
            segments, mesh_vertices, seg_indices, colmap_points
        )
        
        if len(object_points_world) == 0:
            if debug_mode:
                print(f"  SKIPPED: No object points found")
            continue
        
        if debug_mode:
            print(f"  Total object points: {len(object_points_world)}")
            print(f"  Object points range: min={object_points_world.min(axis=0)}, max={object_points_world.max(axis=0)}")
        
        # For large planar objects (walls, floors, ceilings), use more aggressive filtering
        is_large_planar = label.lower() in ['wall', 'floor', 'ceiling', 'ground']
        
        # Project to image with occlusion checking
        points_2d, valid = project_points_to_image(
            object_points_world,
            T_world_cam,
            np.array(camera_info['params']),
            width,
            height,
            camera_model=camera_info.get('model', 'OPENCV'),
            depth_map=depth_map,
            occlusion_threshold=0.1,  # 10cm tolerance for occlusion
            debug=debug_mode
        )
        
        # Get occlusion statistics before filtering
        points_cam_obj = (T_world_cam[:3, :3] @ object_points_world.T).T + T_world_cam[:3, 3]
        # Count how many points would be occluded (approximate - check depth map)
        occlusion_count = 0
        if depth_map is not None and len(object_points_world) > 0:
            # Quick projection to check occlusion
            fx, fy = camera_info['params'][0], camera_info['params'][1]
            cx, cy = camera_info['params'][2], camera_info['params'][3]
            camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
            if camera_info.get('model', 'OPENCV') == 'OPENCV_FISHEYE':
                dist_coeffs = np.array(camera_info['params'][4:8], dtype=np.float32) if len(camera_info['params']) >= 8 else np.zeros(4, dtype=np.float32)
                try:
                    points_2d_check, _ = cv2.fisheye.projectPoints(
                        points_cam_obj.reshape(-1, 1, 3),
                        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                    )
                except:
                    points_2d_check, _ = cv2.projectPoints(
                        points_cam_obj.reshape(-1, 1, 3),
                        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                    )
            else:
                dist_coeffs = np.array(camera_info['params'][4:8], dtype=np.float32) if len(camera_info['params']) >= 8 else np.zeros(4, dtype=np.float32)
                points_2d_check, _ = cv2.projectPoints(
                    points_cam_obj.reshape(-1, 1, 3),
                    np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), camera_matrix, dist_coeffs
                )
            points_2d_check = points_2d_check.reshape(-1, 2)
            x_int_check = np.clip(points_2d_check[:, 0].astype(int), 0, width - 1)
            y_int_check = np.clip(points_2d_check[:, 1].astype(int), 0, height - 1)
            valid_depth_check = (points_cam_obj[:, 2] > 0.2) & (points_cam_obj[:, 2] < 50.0)
            for i in range(len(points_cam_obj)):
                if valid_depth_check[i]:
                    existing_depth = depth_map[y_int_check[i], x_int_check[i]]
                    point_depth = points_cam_obj[i, 2]
                    if existing_depth < np.inf and point_depth > existing_depth + 0.1:
                        occlusion_count += 1
        
        occlusion_ratio = occlusion_count / len(object_points_world) if len(object_points_world) > 0 else 0
        
        if debug_mode:
            print(f"  Points after projection: {np.sum(valid)}/{len(object_points_world)} valid")
            print(f"  Occlusion ratio: {occlusion_ratio:.2%} ({occlusion_count}/{len(object_points_world)} points occluded)")
            if np.any(valid):
                valid_points_2d = points_2d[valid]
                print(f"  Valid points 2D range: x=[{valid_points_2d[:, 0].min():.1f}, {valid_points_2d[:, 0].max():.1f}], y=[{valid_points_2d[:, 1].min():.1f}, {valid_points_2d[:, 1].max():.1f}]")
                valid_indices = np.where(valid)[0]
                depths_obj = points_cam_obj[valid_indices, 2]
                print(f"  Valid points depth range: [{depths_obj.min():.2f}, {depths_obj.max():.2f}] meters")
        
        # AGGRESSIVE FILTERING: If >70% of points are occluded, object is likely not visible
        # Skip entirely or require much stricter conditions
        if occlusion_ratio > 0.70:
            if debug_mode:
                print(f"  SKIPPED: Too occluded ({occlusion_ratio:.1%} > 70%) - object likely not visible")
            continue
        
        if not np.any(valid):
            if debug_mode:
                print(f"  SKIPPED: No valid projections")
            continue
        
        valid_points_2d = points_2d[valid]
        
        # Check valid point ratio - if too few points are valid, object is likely not visible
        valid_point_ratio = len(valid_points_2d) / len(object_points_world) if len(object_points_world) > 0 else 0
        if valid_point_ratio < 0.10:  # Less than 10% of points are valid
            if debug_mode:
                print(f"  SKIPPED: Too few valid points ({valid_point_ratio:.1%} = {len(valid_points_2d)}/{len(object_points_world)}) - object likely not visible")
            continue
        
        # Check depth variance - if depth range is too large, points might be from different parts
        if len(valid_points_2d) > 0:
            points_cam_obj = (T_world_cam[:3, :3] @ object_points_world.T).T + T_world_cam[:3, 3]
            valid_indices = np.where(valid)[0]
            depths_obj = points_cam_obj[valid_indices, 2]
            depth_range = depths_obj.max() - depths_obj.min()
            median_depth = np.median(depths_obj)
            depth_variance_ratio = depth_range / (median_depth + 1e-6)
            
            if debug_mode:
                print(f"  Depth variance: range={depth_range:.2f}m, median={median_depth:.2f}m, ratio={depth_variance_ratio:.2f}")
            
            # If depth variance is >2x median, points are likely from different parts/occluded
            if depth_variance_ratio > 2.0:
                if debug_mode:
                    print(f"  SKIPPED: Excessive depth variance (ratio={depth_variance_ratio:.2f} > 2.0) - points likely from different parts")
                continue
        
        # For objects with moderate occlusion (50-70%), require more valid points
        if occlusion_ratio > 0.50:
            min_points_occluded = max(min_points * 3, 50)  # Require 3x more points or at least 50
            if len(valid_points_2d) < min_points_occluded:
                if debug_mode:
                    print(f"  SKIPPED: High occlusion ({occlusion_ratio:.1%}) but only {len(valid_points_2d)} valid points (need {min_points_occluded})")
                continue
        
        # Update depth map with this object's visible points (for future occlusion checks)
        # This ensures later objects are checked against all previously processed objects
        if len(valid_points_2d) > 0:
            # Get depths for valid points
            points_cam_obj = (T_world_cam[:3, :3] @ object_points_world.T).T + T_world_cam[:3, 3]
            valid_indices = np.where(valid)[0]
            depths_obj = points_cam_obj[valid_indices, 2]
            
            x_int = np.clip(valid_points_2d[:, 0].astype(int), 0, width - 1)
            y_int = np.clip(valid_points_2d[:, 1].astype(int), 0, height - 1)
            
            # Update depth map: keep minimum depth at each pixel
            for i in range(len(x_int)):
                if depths_obj[i] < depth_map[y_int[i], x_int[i]]:
                    depth_map[y_int[i], x_int[i]] = depths_obj[i]
        
        # Visibility check: ensure points are reasonably distributed
        # Objects with all points clustered in a tiny area are likely not actually visible
        # For occluded objects, require better distribution
        if len(valid_points_2d) >= 3:
            point_spread_x = valid_points_2d[:, 0].max() - valid_points_2d[:, 0].min()
            point_spread_y = valid_points_2d[:, 1].max() - valid_points_2d[:, 1].min()
            # Stricter spread requirement for occluded objects
            if occlusion_ratio > 0.50:
                min_spread = min(width, height) * 0.05  # 5% for occluded objects (vs 2% normal)
            else:
                min_spread = min(width, height) * 0.02  # At least 2% of image dimension
            if point_spread_x < min_spread and point_spread_y < min_spread:
                # Points are too clustered - likely invalid projection or occluded object
                if debug_mode:
                    print(f"  SKIPPED: Points too clustered (spread: {point_spread_x:.1f}x{point_spread_y:.1f}, need {min_spread:.1f})")
                continue
            
            # Check for excessive spread - if points span >80% of image, likely a streak
            max_spread_x = width * 0.80  # Max 80% of width
            max_spread_y = height * 0.80  # Max 80% of height
            if point_spread_x > max_spread_x or point_spread_y > max_spread_y:
                if debug_mode:
                    print(f"  SKIPPED: Points span too much of image (spread: {point_spread_x:.1f}x{point_spread_y:.1f}, max: {max_spread_x:.1f}x{max_spread_y:.1f}) - likely streak")
                continue
            
            # Check for excessive spread - if points span >80% of image, likely a streak
            max_spread_x = width * 0.80  # Max 80% of width
            max_spread_y = height * 0.80  # Max 80% of height
            if point_spread_x > max_spread_x or point_spread_y > max_spread_y:
                if debug_mode:
                    print(f"  SKIPPED: Points span too much of image (spread: {point_spread_x:.1f}x{point_spread_y:.1f}, max: {max_spread_x:.1f}x{max_spread_y:.1f}) - likely streak")
                continue
        
        # For large planar objects, apply additional spatial filtering
        if is_large_planar and len(valid_points_2d) > 100:
            # Remove outliers using statistical filtering
            # Points that are too far from the main cluster are likely invalid
            if len(valid_points_2d) > 1:
                # Use median point as reference
                median_point = np.median(valid_points_2d, axis=0)
                distances = np.linalg.norm(valid_points_2d - median_point, axis=1)
                
                # Keep points within 2 standard deviations of median distance
                median_dist = np.median(distances)
                std_dist = np.std(distances)
                threshold_dist = median_dist + 2.5 * std_dist
                inlier_mask = distances < threshold_dist
                
                valid_points_2d = valid_points_2d[inlier_mask]
        
        if len(valid_points_2d) < min_points:
            continue
        
        # Additional visibility check: ensure points are reasonably distributed
        # Objects with all points clustered in a tiny area are likely not actually visible
        if len(valid_points_2d) >= 3:
            # Check point spread - if all points are within a very small area, skip
            point_spread_x = valid_points_2d[:, 0].max() - valid_points_2d[:, 0].min()
            point_spread_y = valid_points_2d[:, 1].max() - valid_points_2d[:, 1].min()
            min_spread = min(width, height) * 0.02  # At least 2% of image dimension
            if point_spread_x < min_spread and point_spread_y < min_spread:
                # Points are too clustered - likely invalid projection or occluded object
                continue
        
        # Create binary mask with more conservative settings for large planar objects or occluded objects
        # For highly occluded objects, disable hole filling and dilation to prevent large masks
        if is_large_planar or occlusion_ratio > 0.50:
            # For walls/floors or occluded objects, use more conservative hole filling and dilation
            binary_mask = points_to_mask(valid_points_2d, width, height, 
                                       fill_holes=False,  # Disable hole filling for occluded objects
                                       dilate=False)  # Disable dilation for occluded objects
        else:
            binary_mask = points_to_mask(valid_points_2d, width, height, fill_holes, dilate)
        
        if debug_mode:
            print(f"  Mask created: {np.sum(binary_mask > 0)} pixels")
        
        # Final visibility check: mask must have reasonable coverage
        # This prevents smearing from objects that aren't actually visible in the frame
        # For occluded objects, use stricter coverage limits
        mask_coverage = np.sum(binary_mask > 0) / (width * height)
        if occlusion_ratio > 0.50:
            # Stricter limits for occluded objects
            min_coverage = 0.0001
            max_coverage = 0.15  # Max 15% for occluded objects (vs 30% normal)
        else:
            min_coverage = 0.0001  # At least 0.01% of image (very small threshold)
            max_coverage = 0.3  # At most 30% of image (more aggressive - prevents smearing)
        
        if debug_mode:
            print(f"  Mask coverage: {mask_coverage:.6f} (min={min_coverage}, max={max_coverage})")
        
        if np.sum(binary_mask > 0) == 0:
            if debug_mode:
                print(f"  SKIPPED: Empty mask")
            continue
        
        # Skip masks that are too small (likely noise) or too large (likely smearing)
        if mask_coverage < min_coverage or mask_coverage > max_coverage:
            if debug_mode:
                print(f"  SKIPPED: Coverage out of range ({mask_coverage:.4f} not in [{min_coverage}, {max_coverage}])")
            continue
        
        # Additional check: mask should have reasonable aspect ratio
        # Smearing often creates very elongated masks
        mask_y, mask_x = np.where(binary_mask > 0)
        if len(mask_x) > 0 and len(mask_y) > 0:
            mask_width = mask_x.max() - mask_x.min() + 1
            mask_height = mask_y.max() - mask_y.min() + 1
            aspect_ratio = max(mask_width, mask_height) / (min(mask_width, mask_height) + 1e-6)
            # Skip masks with extreme aspect ratios (likely smearing)
            if aspect_ratio > 20:  # More than 20:1 aspect ratio is suspicious
                continue
        
        # Save binary mask with object label in filename
        if create_binary:
            # Sanitize label for filename (remove special characters)
            label_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in label)[:30]
            mask_filename = f"object_{object_id:03d}_{label_safe}_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
            mask_path = output_dir / 'binary_masks' / mask_filename
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), binary_mask)
            
            # Create side-by-side visualization with original image
            if original_image is not None:
                # Create overlay: original image with mask in red
                overlay = original_image.copy()
                mask_colored = np.zeros_like(original_image)
                mask_colored[:, :, 2] = binary_mask  # Red channel
                overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
                
                # Create side-by-side: original | overlay | mask
                comparison = np.hstack([
                    original_image,
                    overlay,
                    cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
                ])
                
                label_safe = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in label)[:30]
                comparison_path = output_dir / 'comparisons' / f"object_{object_id:03d}_{label_safe}_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
                comparison_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(comparison_path), comparison)
        
        # Add to instance mask
        if create_instance:
            mask_pixels = binary_mask > 0
            if np.any(mask_pixels):
                instance_mask[mask_pixels] = object_id
        
        image_objects.append({
            'object_id': object_id,
            'label': label,
            'num_points': len(valid_points_2d)
        })
    
    # Save instance mask
    if create_instance and np.any(instance_mask > 0):
        instance_filename = f"instance_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
        instance_path = output_dir / 'instance_masks' / instance_filename
        instance_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(instance_path), instance_mask)
        
        # Visualization version
        instance_vis = instance_mask.copy().astype(np.float32)
        if instance_vis.max() > 0:
            instance_vis = (instance_vis / instance_vis.max() * 255).astype(np.uint8)
        vis_path = output_dir / 'instance_masks' / f"instance_vis_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
        cv2.imwrite(str(vis_path), instance_vis)
        
        # Colored version (improved colormap for better debugging)
        unique_ids = np.unique(instance_mask[instance_mask > 0])
        colored_mask = None
        if len(unique_ids) > 0:
            colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
            # Use a more distinct colormap - cycle through HSV space for better visibility
            for idx, obj_id in enumerate(unique_ids):
                # Distribute hues evenly across 180 degrees for maximum distinction
                hue = int((idx * 180) / max(len(unique_ids), 1))  # 0-180 for OpenCV HSV
                saturation = 255  # Full saturation for vivid colors
                value = 255  # Full brightness
                color_hsv = np.uint8([[[hue, saturation, value]]])
                color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
                colored_mask[instance_mask == obj_id] = color_bgr
            colored_path = output_dir / 'instance_masks' / f"instance_colored_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
            cv2.imwrite(str(colored_path), colored_mask)
        
        # Create side-by-side visualization for instance mask
        if original_image is not None and colored_mask is not None:
            # Create overlay: original image with colored instance mask
            overlay = original_image.copy()
            overlay = cv2.addWeighted(overlay, 0.6, colored_mask, 0.4, 0)
            
            # Create side-by-side: original | overlay | colored_mask
            comparison = np.hstack([
                original_image,
                overlay,
                colored_mask
            ])
            
            comparison_path = output_dir / 'comparisons' / f"instance_{img_name.replace('.jpg', '').replace('.JPG', '')}.png"
            comparison_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(comparison_path), comparison)
    
    return {
        'image_id': img_id,
        'image_name': img_name,
        'objects': image_objects
    }


def main():
    parser = argparse.ArgumentParser(description='Generate instance segmentation masks from annotated PLY')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to ScanNet data directory (e.g., data/00777c41d4)')
    parser.add_argument('--output_dir', type=str, default='output_masks',
                       help='Output directory for masks')
    parser.add_argument('--camera_source', type=str, default='iphone',
                       choices=['iphone', 'dslr'],
                       help='Which camera source to use')
    parser.add_argument('--min_points', type=int, default=5,
                       help='Minimum visible points required per object')
    parser.add_argument('--fill_holes', action='store_true',
                       help='Fill holes in masks')
    parser.add_argument('--dilate', action='store_true',
                       help='Dilate masks slightly')
    parser.add_argument('--binary_only', action='store_true',
                       help='Only create binary masks')
    parser.add_argument('--instance_only', action='store_true',
                       help='Only create instance masks')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load segments_anno.json (handles both ScanNet and ScanNet++ formats)
    segments_anno_path = data_dir / 'scans' / 'segments_anno.json'
    if not segments_anno_path.exists():
        print(f"Error: segments_anno.json not found at {segments_anno_path}")
        return
    
    with open(segments_anno_path, 'r') as f:
        segments_data = json.load(f)
    
    # Handle both formats:
    # - ScanNet: {"segGroups": [...]}
    # - ScanNet++: [...] (list directly)
    if isinstance(segments_data, list):
        objects = segments_data  # ScanNet++ format
    elif isinstance(segments_data, dict) and 'segGroups' in segments_data:
        objects = segments_data['segGroups']  # ScanNet format
    else:
        print("Error: segments_anno.json format not recognized")
        print(f"  Expected either a list or dict with 'segGroups' key")
        return
    
    # Filter out structural elements (walls, floors, ceilings) - not needed for ShapeR
    structural_labels = ['wall', 'floor', 'ceiling', 'ground', 'door', 'window', 'doorframe']
    objects_filtered = []
    for obj in objects:
        label = obj.get('label', 'object').lower()
        if not any(struct_label in label for struct_label in structural_labels):
            objects_filtered.append(obj)
    
    num_excluded = len(objects) - len(objects_filtered)
    objects = objects_filtered
    print(f"Found {len(objects)} objects (excluded {num_excluded} structural elements: walls, floors, ceilings, doors, windows)")
    
    # Load segments.json to map segment IDs to vertex indices
    segments_path = data_dir / 'scans' / 'segments.json'
    if not segments_path.exists():
        print(f"Error: segments.json not found at {segments_path}")
        return
    
    with open(segments_path, 'r') as f:
        segments_data = json.load(f)
    
    if 'segIndices' not in segments_data:
        print("Error: 'segIndices' not found in segments.json")
        return
    
    seg_indices = segments_data['segIndices']  # List of segment IDs per vertex
    print(f"Loaded segment indices for {len(seg_indices)} vertices")
    
    # Load mesh
    mesh_path = data_dir / 'scans' / 'mesh_aligned_0.05.ply'
    if not mesh_path.exists():
        print(f"Error: Mesh file not found at {mesh_path}")
        return
    
    mesh = trimesh.load(str(mesh_path))
    mesh_vertices = np.array(mesh.vertices, dtype=np.float32)
    print(f"Loaded mesh: {len(mesh_vertices)} vertices")
    print(f"Mesh bounds: min={mesh_vertices.min(axis=0)}, max={mesh_vertices.max(axis=0)}")
    
    # Verify segment indices match mesh vertex count
    if len(seg_indices) != len(mesh_vertices):
        print(f"⚠️  WARNING: Segment indices count ({len(seg_indices)}) != mesh vertices ({len(mesh_vertices)})")
        print(f"   This may cause incorrect segmentation. Check if you're using the correct mesh file.")
        # Truncate or pad to match
        if len(seg_indices) > len(mesh_vertices):
            seg_indices = seg_indices[:len(mesh_vertices)]
            print(f"   Truncated seg_indices to match mesh")
        else:
            # Pad with -1 (invalid segment ID)
            seg_indices = seg_indices + [-1] * (len(mesh_vertices) - len(seg_indices))
            print(f"   Padded seg_indices to match mesh")
    
    # Load COLMAP data
    colmap_dir = data_dir / args.camera_source / 'colmap'
    if not colmap_dir.exists():
        print(f"Error: COLMAP directory not found at {colmap_dir}")
        return
    
    print("Loading COLMAP data...")
    cameras, images, points = load_colmap_data(colmap_dir)
    print(f"  Loaded {len(cameras)} cameras, {len(images)} images, {len(points)} points")
    
    # COLMAP is already aligned with the mesh (per ScanNet++ docs)
    # "contains the colmap camera model that has been aligned with the 3D scans"
    # So we do NOT need to align coordinate systems - it would break the alignment!
    if len(points) > 0:
        colmap_points_array = np.array(list(points.values()))
        print(f"  COLMAP point range: min={colmap_points_array.min(axis=0)}, max={colmap_points_array.max(axis=0)}")
        print(f"  ✓ COLMAP is already aligned with mesh (no alignment needed)")
    else:
        print("  Warning: No COLMAP points found")
    
    # Process each image
    create_binary = not args.instance_only
    create_instance = not args.binary_only
    
    all_image_data = []
    successful = 0
    total_masks = 0
    
    # Find RGB image directory
    rgb_dir = None
    if args.camera_source == 'iphone':
        rgb_dir = data_dir / 'iphone' / 'rgb'
    elif args.camera_source == 'dslr':
        rgb_dir = data_dir / 'dslr' / 'resized_images'
    
    if rgb_dir and not rgb_dir.exists():
        print(f"Warning: RGB directory not found at {rgb_dir}, skipping image comparisons")
        rgb_dir = None
    elif rgb_dir:
        print(f"Found RGB images at {rgb_dir}")
    
    # Find depth directory (for occlusion checking)
    depth_dir = None
    if args.camera_source == 'iphone':
        depth_dir = data_dir / 'iphone' / 'depth'
    elif args.camera_source == 'dslr':
        # DSLR might not have depth images
        depth_dir = data_dir / 'dslr' / 'depth'
    
    if depth_dir and not depth_dir.exists():
        print(f"Warning: Depth directory not found at {depth_dir}, occlusion checking will be limited")
        depth_dir = None
    elif depth_dir:
        print(f"Found depth images at {depth_dir}")
    
    print(f"\nGenerating masks for {len(images)} images...")
    for img_id, img_info in images.items():
        result = generate_masks_for_image(
            img_id, img_info, objects,
            mesh_vertices, seg_indices,
            cameras, points, output_dir,
            rgb_dir=rgb_dir,
            depth_dir=depth_dir,
            create_binary=create_binary,
            create_instance=create_instance,
            fill_holes=args.fill_holes,
            dilate=args.dilate,
            min_points=args.min_points
        )
        
        if result and len(result['objects']) > 0:
            all_image_data.append(result)
            successful += 1
            total_masks += len(result['objects'])
            if successful % 100 == 0:
                print(f"  Processed {successful} images, created {total_masks} masks...")
    
    # Save metadata
    metadata = {
        'total_images': len(images),
        'images_with_objects': successful,
        'total_objects': len(objects),
        'total_masks_created': total_masks,
        'object_labels': {obj['id']: obj.get('label', 'object') for obj in objects},
        'images': all_image_data
    }
    
    metadata_path = output_dir / 'mask_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Generated {total_masks} masks across {successful}/{len(images)} images")
    print(f"✓ Saved metadata to {metadata_path}")
    if create_binary:
        print(f"✓ Binary masks: {output_dir / 'binary_masks'}")
    if create_instance:
        print(f"✓ Instance masks: {output_dir / 'instance_masks'}")
    if rgb_dir:
        print(f"✓ Comparison images: {output_dir / 'comparisons'}")


if __name__ == '__main__':
    main()
