import csv
import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
import xml.etree.ElementTree as ET

color_values = [ '-154', '-3611080', '-13210', '-16777088', '-16744193', '-11776948', '-65408',
  '-8388480', '-10040065',    '-39322',  '-1644826', '-13421773', '-16760704',  '-8372224',
 '-15132391',  '-3342490',    '-65536', '-10066177',  '-65281', '-16711936', '-10066330',
 '-16744320', '-16711681', '-12550144',  '-8421505',  '-6710887',  '-8323328',    '-32768',
  '-8388353', '-16776961',  '-3381505', '-10027060',      '-256', '-10027009', '-16744448',
 '-16744384', '-10027162', '-16777216',  '-8388544', '-12582784',    '-36913', '-16711808',
  '-3355444',        '-1']

def plot_image(image, size=30):    
    """
    Plot a grayscale image in the original size - takes time
    """
    fig = plt.figure(figsize=(size, size))
    ax1 = plt.subplot(1, 1, 1) 
    ax1.imshow(image, cmap='gray')  

def xml_to_dataframe(xml_file):
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Initialize lists to store data
    data = []
    columns = ['id', 'name', 'pos_x', 'pos_y', 'color']

    # Iterate through each 'roi' element
    for roi in root.findall('roi'):
        roi_data = {
            'id': roi.find('id').text,
            'name': roi.find('name').text,
            'pos_x': roi.find('position/pos_x').text,
            'pos_y': roi.find('position/pos_y').text,
            'color': roi.find('color').text
        }
        data.append(roi_data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns to appropriate types
    df['id'] = pd.to_numeric(df['id'])
    df['pos_x'] = pd.to_numeric(df['pos_x'])
    df['pos_y'] = pd.to_numeric(df['pos_y'])
    df['color'] = pd.to_numeric(df['color'])

    return df

def dataframe_to_xml(df, filename=None, scale=[1, 1]):
    # Create the root element
    root = ET.Element('root')

    for index, row in df.iterrows():
        roi = ET.SubElement(root, 'roi')
        
        ET.SubElement(roi, 'classname').text = 'plugins.kernel.roi.roi2d.ROI2DPoint'
        ET.SubElement(roi, 'id').text = str(index)  # Increment id for each point
        ET.SubElement(roi, 'name').text = 'Point'+str(index+1)
        ET.SubElement(roi, 'selected').text = 'true'
        ET.SubElement(roi, 'readOnly').text = 'false'
        ET.SubElement(roi, 'properties')
        ET.SubElement(roi, 'color').text = color_values[index % len(color_values)]
        ET.SubElement(roi, 'stroke').text = '2'
        ET.SubElement(roi, 'opacity').text = '0.3'
        ET.SubElement(roi, 'showName').text = 'false'
        ET.SubElement(roi, 'z').text = '-1'
        ET.SubElement(roi, 't').text = '-1'
        ET.SubElement(roi, 'c').text = '-1'
        
        position = ET.SubElement(roi, 'position')
        ET.SubElement(position, 'pos_x').text = str(row['pos_x']*scale[0])
        ET.SubElement(position, 'pos_y').text = str(row['pos_y']*scale[1])

    # Create the XML tree and save to file
    tree = ET.ElementTree(root)
    if filename is not None:
        tree.write(filename, encoding='utf-8', xml_declaration=True)

def dataframe_to_xml_(df, filename=None, scale=[1, 1]):
    # Create the root element
    root = ET.Element('root')

    for index, row in df.iterrows():
        roi = ET.SubElement(root, 'roi')
        
        ET.SubElement(roi, 'classname').text = 'plugins.kernel.roi.roi2d.ROI2DPoint'
        ET.SubElement(roi, 'id').text = str(index)  # Increment id for each point
        ET.SubElement(roi, 'name').text = 'Point'+str(index+1)
        ET.SubElement(roi, 'selected').text = 'true'
        ET.SubElement(roi, 'readOnly').text = 'false'
        ET.SubElement(roi, 'properties')
        ET.SubElement(roi, 'color').text = color_values[index % len(color_values)]# '-20388296' #'-16711936' green, '-65536' red,  '-16776961' blue
        ET.SubElement(roi, 'stroke').text = '2'
        ET.SubElement(roi, 'opacity').text = '0.3'
        ET.SubElement(roi, 'showName').text = 'false'
        ET.SubElement(roi, 'z').text = '-1'
        ET.SubElement(roi, 't').text = '-1'
        ET.SubElement(roi, 'c').text = '-1'
        
        position = ET.SubElement(roi, 'position')
        ET.SubElement(position, 'pos_x').text = str(row['pos_x']*scale[0])
        ET.SubElement(position, 'pos_y').text = str(row['pos_y']*scale[1])

    # Create the XML tree and save to file
    tree = ET.ElementTree(root)
    if filename is not None:
        tree.write(filename, encoding='utf-8', xml_declaration=True)

def list_to_dataframe(coordinates, filename=None):

    # Create a DataFrame
    df = pd.DataFrame(coordinates, columns=['pos_y', 'pos_x'])

    # Add 'id' and 'name' columns
    df['id'] = range(1, len(df) + 1)
    df['name'] = "'Point2D'"
    df = df[['id', 'name', 'pos_x', 'pos_y']]

    if filename is not None:
        df.to_csv(filename, index=False)
    return df

def dataframe_to_nparray(df, scale=[1, 1]):
    return(df[['pos_x', 'pos_y']].values * scale).astype(int)

def dataframe_to_pointcloud(df, filename=None, scale=[1, 1]):
    points2D = (df[['pos_x', 'pos_y']].to_numpy() * scale).astype(int)
    points3D = np.hstack((points2D, np.zeros((len(points2D), 1))))

    # Convert NumPy array to Open3D PointCloud
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points3D)
    
    if filename is not None:
        o3d.io.write_point_cloud(filename, points_pcd)
    return points_pcd

def dataframe_to_csv(df, filename=None, scale=[1, 1]):
    
    # Open a new CSV file in write mode
    with open(filename, 'w', newline='') as csvfile:
    # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
    
    # Write the coordinates
        for coord in (df[['pos_x', 'pos_y']].values * scale).astype(int):
            csv_writer.writerow(coord)

# Visualize results
def visualize_result_nparray(source, target, result, title):
    source_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source)
    source_cloud.paint_uniform_color([1, 0, 0])  # Red

    target_cloud = o3d.geometry.PointCloud()
    target_cloud.points = o3d.utility.Vector3dVector(target)
    target_cloud.paint_uniform_color([0, 1, 0])  # Green

    result_cloud = o3d.geometry.PointCloud()
    result_cloud.points = o3d.utility.Vector3dVector(result)
    result_cloud.paint_uniform_color([0, 0, 1])  # Blue

    o3d.visualization.draw_geometries([source_cloud, target_cloud, result_cloud], window_name=title)

def visualize_result_pcd(source, target, result, title):
    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green
    result.paint_uniform_color([0, 0, 1])  # Blue
    o3d.visualization.draw_geometries([source, target, result], window_name=title)

# Print transformations
def print_transformations(transformation_paramters, title):
    print(title)
    print(transformation_paramters.rot)  # Rotation matrix
    print(transformation_paramters.t)    # Translation vector
    print(transformation_paramters.scale)  # Scale factor

    # Evaluate alignment using chamfer distance
def chamfer_distance(A, B, title):
    distances = np.min(np.sum((A[:, np.newaxis, :] - B[np.newaxis, :, :]) ** 2, axis=2), axis=1)
    print(f"Chamfer distance = {np.mean(distances)} ({title}) ")

    return np.mean(distances)

def convert_to_pcd(nparray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(nparray)
    return pcd

def save_correspondences_in2df(source, target, correspondences):
    # create empty dataframe
    source_df = pd.DataFrame()
    target_df = pd.DataFrame()

    # add columns to the dataframe ['id', 'name', 'source_pos_x', 'source_pos_y', target_pos_x, target_pos_y] and the length of the dataframe is the number of correspondences
    source_df['id'] = range(1, len(correspondences) + 1)
    target_df['id'] = range(1, len(correspondences) + 1)
    source_df['name'] = "'Point2D'"
    target_df['name'] = "'Point2D'"
    source_df['pos_x'] = source[correspondences[:, 0], 0]
    source_df['pos_y'] = source[correspondences[:, 0], 1]
    target_df['pos_x'] = target[correspondences[:, 1], 0]
    target_df['pos_y'] = target[correspondences[:, 1], 1]
    
    return source_df, target_df


def save_correspondences_in1df(source, target, correspondences):
    # create empty dataframe
    df = pd.DataFrame()

    # add columns to the dataframe ['id', 'name', 'source_pos_x', 'source_pos_y', target_pos_x, target_pos_y] and the length of the dataframe is the number of correspondences
    df['id'] = range(1, len(correspondences) + 1)
    df['name'] = "'Point2D'"
    df['source_pos_x'] = source[correspondences[:, 0], 0]
    df['target_pos_x'] = target[correspondences[:, 1], 0]
    df['source_pos_y'] = source[correspondences[:, 0], 1]
    df['target_pos_y'] = target[correspondences[:, 1], 1]
    df['source_ind'] = correspondences[:, 0]
    df['target_ind'] = correspondences[:, 1]
    df['distance'] = np.sqrt(np.sum((source[correspondences[:, 0]] - target[correspondences[:, 1]]) ** 2, axis=1))

    return df

def clean_correspondences(correspondences):
    # Get unique target indices and their first occurrences
    unique_target_indices, unique_indices = np.unique(correspondences[:, 1], return_index=True)

    # Use the unique_indices to select the rows from the original array
    cleaned_correspondences = correspondences[unique_indices]

    return cleaned_correspondences

def euclidean_distance(df1, df2):
    dist = np.sqrt((df1['pos_x'] - df2['pos_x'])**2 + (df1['pos_y'] - df2['pos_y'])**2)
    return dist

def euclidean_distance(df):
    dist = np.sqrt((df['source_pos_x'] - df['target_pos_x'])**2 + (df['source_pos_y'] - df['target_pos_y'])**2)
    return dist


def clean_close_points(points, threshold=10):
    """
    Remove points closer than `threshold` units to any previously kept point.
    Points are processed in order, and earlier points are prioritized.
    """
    filtered = []
    for p in points:
        if all(np.linalg.norm(p - fp) >= threshold for fp in filtered):
            filtered.append(p)
    return np.array(filtered)

def refine_to_local_maxima(coords, image, window_size=5):
    """
    Refine coordinates to the nearest local maximum in the image.
    
    Args:
        coords: (y, x) coordinates to refine (shape: Nx2)
        image: 2D image array
        window_size: Size of neighborhood for local maxima detection (odd integer)
    
    Returns:
        Refined coordinates as a NumPy array.
    """
    # Find local maxima in the entire image
    local_max_mask = (image == maximum_filter(image, size=window_size))
    
    refined_coords = []
    for y, x in coords:
        # Define neighborhood bounds
        y_min = max(y - window_size//2, 0)
        y_max = min(y + window_size//2 + 1, image.shape[0])
        x_min = max(x - window_size//2, 0)
        x_max = min(x + window_size//2 + 1, image.shape[1])
        
        # Extract neighborhood and local maxima within it
        neighborhood_max_mask = local_max_mask[y_min:y_max, x_min:x_max]
        local_max_coords = np.argwhere(neighborhood_max_mask)
        
        if local_max_coords.size == 0:
            refined_coords.append([y, x])  # No local max found
            continue
        
        # Find closest local max to original coordinate
        distances = np.linalg.norm(
            local_max_coords - np.array([y - y_min, x - x_min]),
            axis=1
        )
        closest_idx = np.argmin(distances)
        local_max_y, local_max_x = local_max_coords[closest_idx]
        
        refined_coords.append([
            y_min + local_max_y,
            x_min + local_max_x
        ])
    
    return np.array(refined_coords)
