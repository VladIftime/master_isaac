import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import carb
from typing import Optional
import os


def save_point_cloud_as_png(point_cloud, filename, projection="xy"):
    """
    Save a 2D projection of the point cloud as a .png image.
    :param point_cloud: Nx3 array of points (x, y, z).
    :param filename: Output file name.
    :param projection: Projection plane ('xy', 'xz', or 'yz').
    """
    # Ensure the point cloud is Nx3
    assert point_cloud.shape[1] == 3, "Point cloud must be Nx3"

    # Select the projection plane
    if projection == "xy":
        x, y = point_cloud[:, 0], point_cloud[:, 1]
        xlabel, ylabel = "X", "Y"
    elif projection == "xz":
        x, y = point_cloud[:, 0], point_cloud[:, 2]
        xlabel, ylabel = "X", "Z"
    elif projection == "yz":
        x, y = point_cloud[:, 1], point_cloud[:, 2]
        xlabel, ylabel = "Y", "Z"
    else:
        raise ValueError("Invalid projection. Use 'xy', 'xz', or 'yz'.")

    # Create the scatter plot
    plt.figure(figsize=(10, 10))
    plt.scatter(
        x, y, s=1, c=point_cloud[:, 2], cmap="viridis"
    )  # Color by Z-axis for depth
    plt.colorbar(label="Depth (Z)")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"2D Projection of Point Cloud ({projection.upper()} Plane)")
    plt.grid(True)

    # Save the plot as a .png file
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


def depth_image_from_distance_image(distance, intrinsics):
    """Computes depth image from distance image.

    Background pixels have depth of 0

    Args:
        distance: HxW float array (meters)
        intrinsics: 3x3 float array

    Returns:
        z: HxW float array (meters)

    """
    fx = intrinsics[0][0]
    cx = intrinsics[0][2]
    fy = intrinsics[1][1]
    cy = intrinsics[1][2]

    height, width = distance.shape
    xlin = np.linspace(0, width - 1, width)
    ylin = np.linspace(0, height - 1, height)
    px, py = np.meshgrid(xlin, ylin)

    x_over_z = (px - cx) / fx
    y_over_z = (py - cy) / fy

    # Compute depth
    z = distance / np.sqrt(1.0 + x_over_z**2 + y_over_z**2)

    # Handle background pixels
    # Assuming background pixels in the distance image are represented by a large value (e.g., infinity)
    # get the largest value in the distance image
    background_value = np.max(distance)
    z[distance == background_value] = 0

    return z


def get_pointcloud(color_img, distance_img, cam_intrinsics, is_orthographic=True):
    """Convert RGB-D images to a 3D point cloud.

    Args:
        color_img: RGB image (HxWx3)
        distance_img: Distance image (HxW) or depth image
        cam_intrinsics: Camera intrinsics matrix (3x3)
        is_orthographic: Whether the camera uses orthographic projection

    Returns:
        points: Nx3 array of 3D points
        colors: Nx3 array of RGB colors
    """
    # Get image dimensions
    im_h = distance_img.shape[0]
    im_w = distance_img.shape[1]

    # Convert distance to depth if needed
    depth_img = depth_image_from_distance_image(distance_img, cam_intrinsics)

    # Project depth into 3D point cloud in camera coordinates
    pix_x, pix_y = np.meshgrid(
        np.linspace(0, im_w - 1, im_w), np.linspace(0, im_h - 1, im_h)
    )

    if is_orthographic:
        # For orthographic projection, the x and y coordinates are directly proportional
        # to the pixel coordinates, and don't depend on depth
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]

        # Calculate world coordinates (flat plane, no perspective)
        cam_pts_x = (
            pix_x - cx
        ) / fx  # Scale by focal length but don't multiply by depth
        cam_pts_y = (pix_y - cy) / fy
        cam_pts_z = depth_img.copy()  # Z is just the depth
    else:
        # For perspective projection
        cam_pts_x = np.multiply(
            pix_x - cam_intrinsics[0][2], depth_img / cam_intrinsics[0][0]
        )
        cam_pts_y = np.multiply(
            pix_y - cam_intrinsics[1][2], depth_img / cam_intrinsics[1][1]
        )
        cam_pts_z = depth_img.copy()

    # Reshape for output
    cam_pts_x = cam_pts_x.reshape(-1, 1)
    cam_pts_y = cam_pts_y.reshape(-1, 1)
    cam_pts_z = cam_pts_z.reshape(-1, 1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:, :, 0]
    rgb_pts_g = color_img[:, :, 1]
    rgb_pts_b = color_img[:, :, 2]
    rgb_pts_r = rgb_pts_r.reshape(-1, 1)
    rgb_pts_g = rgb_pts_g.reshape(-1, 1)
    rgb_pts_b = rgb_pts_b.reshape(-1, 1)

    # Combine points and colors
    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    # Filter out points with zero or invalid depth
    valid_depth = (cam_pts_z.reshape(-1) > 0) & np.isfinite(cam_pts_z.reshape(-1))
    cam_pts = cam_pts[valid_depth]
    rgb_pts = rgb_pts[valid_depth]

    # Debug: Save raw point cloud before any transformations
    plt.figure(figsize=(10, 10))
    plt.scatter(cam_pts[:, 0], cam_pts[:, 1], c=cam_pts[:, 2], cmap="viridis", s=1)
    plt.colorbar(label="Depth (Z)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Raw Point Cloud (Camera Frame)")
    plt.savefig("raw_point_cloud.png")
    plt.close()

    # Debug: 3D visualization of raw point cloud
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    # Sample points to avoid overcrowding
    sample_size = min(5000, cam_pts.shape[0])
    if sample_size > 0:
        sample_indices = np.random.choice(cam_pts.shape[0], sample_size, replace=False)
        ax.scatter(
            cam_pts[sample_indices, 0],
            cam_pts[sample_indices, 1],
            cam_pts[sample_indices, 2],
            c=cam_pts[sample_indices, 2],
            cmap="viridis",
            s=5,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Raw Point Cloud (Camera Frame)")
    plt.savefig("3d_raw_point_cloud.png")
    plt.close()

    return cam_pts, rgb_pts


def get_heightmap(
    color_img,
    distance_img,
    pointcloud,
    cam_intrinsics,
    cam_pose,
    workspace_limits,
    heightmap_resolution,
    is_orthographic=True,
    show_heightmap=False,
):
    """Generate a heightmap from RGB-D images.

    Args:
        color_img: RGB image (HxWx3)
        distance_img: Distance image (HxW)
        pointcloud: Point cloud data (Nx3)
        cam_intrinsics: Camera intrinsics matrix (3x3)
        cam_pose: Camera pose as a 4x4 transformation matrix or a tuple of (position, orientation)
        workspace_limits: Workspace limits (ignored, using point cloud bounds instead)
        heightmap_resolution: Resolution of the heightmap in meters
        is_orthographic: Whether the camera uses orthographic projection

    Returns:
        depth_heightmap: Depth heightmap (HxW)
    """
    # Ensure cam_pose is a numpy array
    cam_pose = np.array(cam_pose)
    # carb.log_warn(f"Camera pose matrix:\n{cam_pose}")
    # carb.log_warn(f"Camera intrinsics matrix:\n{cam_intrinsics}")

    # Transform 3D point cloud from camera coordinates to robot coordinates
    pointcloud_robot = pointcloud

    # Get point cloud bounds
    x_min, x_max = np.min(pointcloud_robot[:, 0]), np.max(pointcloud_robot[:, 0])
    y_min, y_max = np.min(pointcloud_robot[:, 1]), np.max(pointcloud_robot[:, 1])

    # Create grid coordinates based on point cloud bounds
    x_coords = np.arange(x_min, x_max + heightmap_resolution, heightmap_resolution)
    y_coords = np.arange(y_min, y_max + heightmap_resolution, heightmap_resolution)

    # Create meshgrid for the XY plane
    xx, yy = np.meshgrid(x_coords, y_coords)
    heightmap = np.zeros_like(xx)

    # For each point in the point cloud, find its corresponding cell in the grid
    x_indices = ((pointcloud_robot[:, 0] - x_min) / heightmap_resolution).astype(int)
    y_indices = ((pointcloud_robot[:, 1] - y_min) / heightmap_resolution).astype(int)

    # For each valid cell, take the maximum z-value of points that fall into it
    for i in range(len(pointcloud_robot)):
        if 0 <= x_indices[i] < len(x_coords) and 0 <= y_indices[i] < len(y_coords):
            current_height = heightmap[y_indices[i], x_indices[i]]
            if current_height == 0 or pointcloud_robot[i, 2] > current_height:
                heightmap[y_indices[i], x_indices[i]] = pointcloud_robot[i, 2]

    # Display the projected heightmap
    if show_heightmap:
        plt.figure(figsize=(10, 10))
        plt.imshow(heightmap, cmap="viridis", origin="lower")
        plt.colorbar(label="Height (Z)")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title("Projected Heightmap")
        plt.show()

    # Create point cloud representation of the heightmap
    non_zero_mask = heightmap != 0
    projected_points = np.zeros((np.sum(non_zero_mask), 3))
    projected_points[:, 0] = xx[non_zero_mask]
    projected_points[:, 1] = yy[non_zero_mask]
    projected_points[:, 2] = heightmap[non_zero_mask]

    return projected_points


def display_heightmap(heightmap, name: Optional[str] = None):
    plt.figure(figsize=(10, 10))
    plt.scatter(
        heightmap[:, 0], heightmap[:, 1], c=heightmap[:, 2], cmap="viridis", s=1
    )
    plt.colorbar(label="Height (Z)")
    plt.xlabel("X")
    plt.ylabel("Y")
    if name is not None:
        plt.title(name)
    plt.show()


def get_height_at_position(heightmap_points, query_x, query_y, radius=0.01):
    """Get the height (z-value) at a specific x,y coordinate from a heightmap point cloud.

    Args:
        heightmap_points: Nx3 array of points from the heightmap
        query_x: x-coordinate to query
        query_y: y-coordinate to query
        radius: radius to search for nearby points (in same units as heightmap)

    Returns:
        float: Interpolated height at the query point. Returns None if no points found within radius.
    """
    # Calculate distances from query point to all heightmap points
    distances = np.sqrt(
        (heightmap_points[:, 0] - query_x) ** 2
        + (heightmap_points[:, 1] - query_y) ** 2
    )

    # Find points within the radius
    nearby_points = heightmap_points[distances < radius]

    if len(nearby_points) == 0:
        return None

    # Weight points by inverse distance
    weights = 1 / (
        distances[distances < radius] + 1e-6
    )  # Add small epsilon to avoid division by zero
    weights = weights / np.sum(weights)  # Normalize weights

    # Calculate weighted average height
    interpolated_height = np.sum(nearby_points[:, 2] * weights)

    return interpolated_height
