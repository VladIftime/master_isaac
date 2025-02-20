import numpy as np
import matplotlib.pyplot as plt


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


def get_pointcloud(color_img, depth_img, cam_intrinsics):
    """Convert RGB-D images to a 3D point cloud."""
    # Get image dimensions
    im_h, im_w = depth_img.shape

    # Create a meshgrid of pixel coordinates
    pix_x, pix_y = np.meshgrid(np.arange(im_w), np.arange(im_h))

    # Convert pixel coordinates to normalized image coordinates
    cam_x = (pix_x - cam_intrinsics[0, 2]) / cam_intrinsics[0, 0]
    cam_y = (pix_y - cam_intrinsics[1, 2]) / cam_intrinsics[1, 1]

    # Reconstruct 3D coordinates from depth image
    cam_z = depth_img
    cam_x = np.multiply(cam_x, cam_z)
    cam_y = np.multiply(cam_y, cam_z)

    # Stack to form a 3D point cloud make the shape of the point cloud Nx3
    # Make sure the projection is correct
    points_xy = np.stack((cam_x, cam_y, cam_z), axis=-1).reshape(-1, 3)
    colors = color_img.reshape(-1, 3)


    return points_xy, colors


def get_heightmap(
    color_img,
    depth_img,
    cam_intrinsics,
    cam_pose,
    workspace_limits,
    heightmap_resolution,
):
    """Generate a heightmap from RGB-D images."""

    # Compute heightmap size
    heightmap_size = np.round(
        (
            (workspace_limits[1][1] - workspace_limits[1][0]) / heightmap_resolution,
            (workspace_limits[0][1] - workspace_limits[0][0]) / heightmap_resolution,
        )
    ).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(
        np.dot(cam_pose[0:3, 0:3], np.transpose(surface_pts))
        + np.tile(cam_pose[0:3, 3:], (1, surface_pts.shape[0]))
    )
    #visualize the surface_pts
    save_point_cloud_as_png(surface_pts, "surface_pts.png")
    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:, 2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]

    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(
        np.logical_and(
            np.logical_and(
                np.logical_and(
                    surface_pts[:, 0] >= workspace_limits[0][0],
                    surface_pts[:, 0] < workspace_limits[0][1],
                ),
                surface_pts[:, 1] >= workspace_limits[1][0],
            ),
            surface_pts[:, 1] < workspace_limits[1][1],
        ),
        surface_pts[:, 2] < workspace_limits[2][1],
    )
    print(f"heightmap_valid_ind: {heightmap_valid_ind}")
    surface_pts = surface_pts[heightmap_valid_ind]
    color_pts = color_pts[heightmap_valid_ind]
    save_point_cloud_as_png(surface_pts, "surface_pts_filtered.png")
    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    color_heightmap_g = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    color_heightmap_b = np.zeros(
        (heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8
    )
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor(
        (surface_pts[:, 0] - workspace_limits[0][0]) / heightmap_resolution
    ).astype(int)
    heightmap_pix_y = np.floor(
        (surface_pts[:, 1] - workspace_limits[1][0]) / heightmap_resolution
    ).astype(int)
    color_heightmap_r[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [0]]
    color_heightmap_g[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [1]]
    color_heightmap_b[heightmap_pix_y, heightmap_pix_x] = color_pts[:, [2]]
    color_heightmap = np.concatenate(
        (color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2
    )
    depth_heightmap[heightmap_pix_y, heightmap_pix_x] = surface_pts[:, 2]
    z_bottom = workspace_limits[2][0]

    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap
