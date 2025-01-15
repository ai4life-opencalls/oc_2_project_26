import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

def plot_image(image, size):    
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
    columns = ['id', 'name', 'pos_x', 'pos_y']

    # Iterate through each 'roi' element
    for roi in root.findall('roi'):
        roi_data = {
            'id': roi.find('id').text,
            'name': roi.find('name').text,
            'pos_x': roi.find('position/pos_y').text,
            'pos_y': roi.find('position/pos_x').text
        }
        data.append(roi_data)

    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Convert numeric columns to appropriate types
    df['id'] = pd.to_numeric(df['id'])
    df['pos_x'] = pd.to_numeric(df['pos_x'])
    df['pos_y'] = pd.to_numeric(df['pos_y'])

    return df

def dataframe_to_xml(df, filename=None):
    # Create the root element
    root = ET.Element('root')

    for index, row in df.iterrows():
        roi = ET.SubElement(root, 'roi')
        
        ET.SubElement(roi, 'classname').text = 'plugins.kernel.roi.roi2d.ROI2DPoint'
        ET.SubElement(roi, 'id').text = str(index)  # Increment id for each point
        ET.SubElement(roi, 'name').text = 'Point2D'
        ET.SubElement(roi, 'selected').text = 'true'
        ET.SubElement(roi, 'readOnly').text = 'false'
        ET.SubElement(roi, 'properties')
        ET.SubElement(roi, 'color').text = '-20388296' #'-16711936' green, '-65536' red,  '-16776961' blue
        ET.SubElement(roi, 'stroke').text = '2'
        ET.SubElement(roi, 'opacity').text = '0.3'
        ET.SubElement(roi, 'showName').text = 'false'
        ET.SubElement(roi, 'z').text = '-1'
        ET.SubElement(roi, 't').text = '-1'
        ET.SubElement(roi, 'c').text = '-1'
        
        position = ET.SubElement(roi, 'position')
        ET.SubElement(position, 'pos_x').text = str(row['pos_x'])
        ET.SubElement(position, 'pos_y').text = str(row['pos_y'])

    # Create the XML tree and save to file
    tree = ET.ElementTree(root)
    if filename is not None:
        tree.write(filename, encoding='utf-8', xml_declaration=True)

def list_to_dataframe(coordinates, filename=None):

    # Create a DataFrame
    df = pd.DataFrame(coordinates, columns=['pos_x', 'pos_y'])

    # Add 'id' and 'name' columns
    df['id'] = range(1, len(df) + 1)
    df['name'] = "'Point2D'"
    df = df[['id', 'name', 'pos_x', 'pos_y']]

    if filename is not None:
        df.to_csv(filename, index=False)
    return df

def dataframe_to_nparray(df):
    return df[['pos_x', 'pos_y']].values

def dataframe_to_pointcloud(df, filename=None):
    points2D = df[['pos_x', 'pos_y']].values
    points3D = np.hstack((points2D, np.zeros((len(points2D), 1))))

    # Convert NumPy array to Open3D PointCloud
    points_pcd = o3d.geometry.PointCloud()
    points_pcd.points = o3d.utility.Vector3dVector(points3D)
    
    if filename is not None:
        o3d.io.write_point_cloud(filename, points_pcd)
    return points_pcd
