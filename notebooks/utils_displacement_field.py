import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from flow_vis import flow_vis
from scipy.interpolate import interpn

def expand_displacement_field(points, displacements, image_shape):
    # Create a grid covering the entire image
    y, x = np.mgrid[0:image_shape[0], 0:image_shape[1]]   # y, x = np.mgrid[0:image_shape[0], 0:image_shape[1]]

    # Interpolate
    expanded_field = griddata(points, displacements, (y, x), method='linear', fill_value=0)  #
 
 #   # Interpolate x and y displacements separately
 #   dx = griddata(points, displacements[:, 0] * weights, (y, x), method='linear', fill_value=0)
 #   dy = griddata(points, displacements[:, 1] * weights, (y, x), method='linear', fill_value=0)
 #   
 #   # Stack the two displacement fields
 #   expanded_field = np.stack([dx, dy], axis=-1)
    return expanded_field

def calculate_displacement_vectors(source_database, target_database, scale_x, scale_y):
    displacements_x = np.asarray((np.asarray(source_database['pos_x']*scale_y) - target_database['pos_x']))
    displacements_y = np.asarray((np.asarray(source_database['pos_y']*scale_x) - target_database['pos_y']))
    displacements = np.column_stack((displacements_x, displacements_y))
    return displacements

def visualize_extended_field(expanded_field, point_cloud):
    flow_color = flow_vis.flow_to_color(expanded_field, convert_to_bgr=False)
    plt.imshow(flow_color)
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c='r', s=5)     # plt.scatter(point_cloud[:, 1], point_cloud[:, 0], c='r', s=5)
    plt.show()

def extrapolate_displacement_field(expanded_field):
    H, W, _ = expanded_field.shape

    # Identify valid (non-zero) displacement vectors
    # This assumes [0, 0] is only used outside the convex hull
    known_mask = np.any(expanded_field != 0, axis=-1)

    # Initialize output with the original field
    extrapolated_field = expanded_field.copy()

    # Use distance transform to get the nearest known value for each pixel
    distance, indices = distance_transform_edt(~known_mask, return_indices=True)

    # Fill zero values with the displacement from the nearest known pixel
    for i in range(2):  # for x and y components
        channel = expanded_field[..., i]
        filled = channel[indices[0], indices[1]]
        extrapolated_field[..., i] = np.where(known_mask, channel, filled)

    return extrapolated_field

def plot_and_save_overlay_images(EMimage, warped_LMimage, filename):
    plt.figure(figsize=(8, 27))
    plt.imshow(EMimage, cmap='gray')              # Base image (e.g., grayscale)
    plt.imshow(warped_LMimage, alpha=0.5, cmap='jet')    # Overlay with 50% transparency; adjust cmap as needed

    plt.axis('off')
    plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)  # 
    plt.show()

def warp_image(image, displacement_field):
    height, width = image.shape[:2]
    y_coords, x_coords = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    src_coords = np.stack([
        y_coords + displacement_field[:,:,1], 
        x_coords + displacement_field[:,:,0]
    ], axis=-1)
    
    warped_image = interpn(
        (np.arange(height), np.arange(width)),
        image,
        src_coords,
        method='linear',
        bounds_error=False,
        fill_value=0
    )
    return warped_image.astype(image.dtype)