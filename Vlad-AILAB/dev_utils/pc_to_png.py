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
