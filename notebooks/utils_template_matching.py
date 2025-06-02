import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_iou(box, boxes):
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Calculate union areas
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    return iou

def non_max_suppression(boxes, scores, threshold):
    # Sort boxes by score in descending order
    sorted_indices = np.argsort(scores)[::-1]
    # print("Indices: ", sorted_indices)
    
    keep_boxes = []
    
    while sorted_indices.size > 0:
        # Pick the box with the highest score
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        
        # Calculate IoU of the picked box with the rest
        ious = calculate_iou(boxes[box_id], boxes[sorted_indices[1:]])
        
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < threshold)[0]
        
        # Update the indices
        sorted_indices = sorted_indices[keep_indices + 1]
    print("Keep boxes: ", keep_boxes)
    return keep_boxes

def template_matching(image, template, threshold):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    #plot_image(result)
    locations = np.where(result >= threshold)
    scores = result[locations]
    matches = list(zip(*locations[::-1]))

    return matches, scores

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Filtering for template matching results:
def otsu_threshold_1d(data, nbins=100):
    # Compute histogram and bin edges
    hist, bin_edges = np.histogram(data, bins=nbins)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Normalize histogram to get probabilities
    hist = hist.astype(float)
    total = hist.sum()
    prob = hist / total

    # Cumulative sums and means
    cumulative_prob = np.cumsum(prob)
    cumulative_mean = np.cumsum(prob * bin_mids)
    global_mean = cumulative_mean[-1]

    # Compute between-class variance for all thresholds
    numerator = (global_mean * cumulative_prob - cumulative_mean) ** 2
    denominator = cumulative_prob * (1 - cumulative_prob)
    # Avoid division by zero
    denominator[denominator == 0] = 1e-10
    sigma_b_squared = numerator / denominator

    # Find the threshold with the maximum between-class variance
    idx = np.argmax(sigma_b_squared)
    threshold = bin_mids[idx]
    return threshold

def average_perimeter_intensity(image, center, radius):
    # Create a circular mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, 1)
    
    # Extract perimeter pixels
    perimeter_pixels = image[mask == 255]
    
    # Calculate average intensity
    average_intensity = np.mean(perimeter_pixels)
    #print("Average perimeter intensity: ", average_intensity)
    
    return average_intensity

def average_intensity_square(image, middle, size):
    x, y = middle
    square = image[y-size:y+size+1, x-size:x+size+1].copy()
    return np.mean(square)

def average_intensity_square_ring(image, middle, out_size, in_size):
    x, y = middle
    square = image[y-out_size:y+out_size+1, x-out_size:x+out_size+1].copy()
    square[in_size:-in_size,in_size:-in_size]=0
    return np.average(square[square!=0])

def filter_the_template_matching_results(locs, image, size, out_size, in_size):
    intensities_middle = []
    intensities_square_ring = []
    for loc in locs:
        intensities_middle.append(average_intensity_square(image, loc, size)) 
        intensities_square_ring.append(average_intensity_square_ring(image, loc, out_size, in_size))

    thresh_middle = otsu_threshold_1d(intensities_middle, 70)
    thresh_square_ring = otsu_threshold_1d(intensities_square_ring, 70)

    fig, axes = plt.subplots(1, 2, figsize=(15, 10), layout='constrained')
    axes[0].hist(intensities_middle, bins=100, color='blue', edgecolor='black')
    axes[0].set_title('Histogram of Intensities Middle')
    axes[0].set_xlabel('Intensity')
    axes[0].set_ylabel('Frequency')

    # Plot histogram for intensities_square_ring on the second subplot
    axes[1].hist(intensities_square_ring, bins=100, color='green', edgecolor='black')
    axes[1].set_title('Histogram of Intensities Square Ring')
    axes[1].set_xlabel('Intensity')
    axes[1].set_ylabel('Frequency')

    print("Threshold middle: ", thresh_middle)
    print("Threshold square ring: ", thresh_square_ring)

    locs1 = [loc for loc in locs if average_intensity_square(image, loc, size) < thresh_middle]                # test 1
    locs2 = [loc for loc in locs if average_intensity_square_ring(image, loc, out_size, in_size) < thresh_square_ring]   # test 2
    locs3 = [loc for loc in locs1 if average_intensity_square_ring(image, loc, out_size, in_size) < thresh_square_ring]  # only if passed both tests
    return locs1, locs2, locs3

def detect_regions(img_mask, min_area=0):
    """
    Detects regions in a binary mask image and returns their centroids and areas.
    
    Parameters:
    img_mask (numpy.ndarray): Binary mask image.
    min_area (int): Minimum area of the regions to be considered.
    
    Returns:
    list: List of tuples containing the centroid coordinates and area of each region.
    """
    # Find connected components
    _, labels = cv2.connectedComponents(img_mask)

    # Calculate the centroid and area of each connected component
    centroids = []
    areas = []

    for label in np.unique(labels):
        if label == 0:             # skip the background
            continue
        
        mask = np.zeros_like(img_mask, dtype=np.uint8)
        mask[labels == label] = 1
        
        moments = cv2.moments(mask)

        if moments["m00"] > min_area :   # equivalent to the area of 3 fiducial particles
            centroids.append((int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])))
            areas.append(int(moments["m00"]))

    return centroids, areas

def draw_red_circles(image_path, centroids):
    """
    Draws circles around the detected regions in the mask image.
    
    Parameters:
    img_mask (numpy.ndarray): Binary mask image.
    centroids (list): List of tuples containing the centroid coordinates of each region.
    
    Returns:
    numpy.ndarray: Image with drawn regions.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    for centroid in centroids:
        cv2.circle(img_color, centroid, 5, ( 0, 0, 255), -1)

    return img_color