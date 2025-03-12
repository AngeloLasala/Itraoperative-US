import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial import KDTree
from scipy import ndimage
import torch

def generate_2d_gaussian(shape, center=None, sigma=1.0):
    """
    Generate a 2D Gaussian distribution.
    
    Parameters:
    shape : tuple of ints
        The shape of the output array (rows, cols).
    center : tuple of floats, optional
        The center (x, y) of the Gaussian. Default is the center of the array.
    sigma : float, optional
        The standard deviation of the Gaussian.
    
    Returns:
    numpy.ndarray
        A 2D array with the Gaussian distribution normalized to integrate to 1.
    """
    rows, cols = shape
    if center is None:
        x0 = (cols - 1) / 2.0
        y0 = (rows - 1) / 2.0
    else:
        x0, y0 = center
    
    # Generate grid of coordinates
    x, y = np.arange(cols), np.arange(rows)
    X, Y = np.meshgrid(x, y)
    
    # Compute Gaussian
    exponent = -((X - x0)**2 + (Y - y0)**2) / (2.0 * sigma**2)
    gaussian = np.exp(exponent) / (2 * np.pi * sigma**2)
    
    return gaussian

def generate_circular_gaussian(shape, center=None, R=10.0, sigma=1.0):
    """
    Generate a 2D Gaussian distribution peaking along a circle of radius R.
    
    Parameters:
    shape : tuple of ints
        Output array shape (rows, cols).
    center : tuple of floats, optional
        Center (x, y) of the circle. Defaults to array center.
    R : float, optional
        Radius of the circle where Gaussian peaks.
    sigma : float, optional
        Controls the width of the Gaussian around the circle.
    
    Returns:
    numpy.ndarray
        2D array with values peaking at 1 along the circle.
    """
    rows, cols = shape
    if center is None:
        x0 = (cols - 1) / 2.0
        y0 = (rows - 1) / 2.0
    else:
        x0, y0 = center
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Compute distance from center
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    # Gaussian centered at distance R
    exponent = -((r - R)**2) / (2 * sigma**2)
    gaussian = np.exp(exponent)
    
    return gaussian

def generate_mexican_hat(shape, center=None, R=10.0, sigma_in=1.0, sigma_out=5.0):
    """
    Generate a ring-shaped Mexican hat profile (Difference of Gaussians).
    
    Parameters:
    shape : tuple
        Output array shape (rows, cols).
    center : tuple, optional
        Center (x, y) of the circle. Defaults to array center.
    R : float, optional
        Radius of the circle where the profile peaks.
    sigma_in : float, optional
        Inner Gaussian width (controls negative lobes).
    sigma_out : float, optional
        Outer Gaussian width (controls positive ring).
    
    Returns:
    numpy.ndarray
        2D array with positive peaks at radius R and negative lobes inside/outside.
    """
    rows, cols = shape
    if center is None:
        x0 = (cols - 1) / 2.0
        y0 = (rows - 1) / 2.0
    else:
        x0, y0 = center
    
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    r = np.sqrt((X - x0)**2 + (Y - y0)**2)
    
    # Generate two Gaussians and subtract them
    gaussian_in = np.exp(-((r - R)**2) / (2 * sigma_in**2))
    gaussian_out = np.exp(-((r - R)**2) / (2 * sigma_out**2))
    mexican_hat = gaussian_out - gaussian_in  # Creates negative lobes
    
    # Normalize to ensure max value is 1
    mexican_hat /= mexican_hat.max()
    
    return mexican_hat

def generate_square_edge_gaussian(shape, center=None, half_size=10.0, sigma=1.0):
    """
    Generate a 2D Gaussian heatmap peaking at the edges of a square.
    
    Parameters:
    shape : tuple of ints
        Output array shape (rows, cols).
    center : tuple of floats, optional
        Center (x, y) of the square. Defaults to array center.
    half_size : float, optional
        Half the side length of the square (distance from center to edge).
    sigma : float, optional
        Controls the width of the Gaussian decay from the edges.
    
    Returns:
    numpy.ndarray
        2D array with maximum values (1) along the square's edges.
    """
    rows, cols = shape
    if center is None:
        x0 = (cols - 1) / 2.0
        y0 = (rows - 1) / 2.0
    else:
        x0, y0 = center
    
    # Create coordinate grid
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    
    # Compute distance to nearest edge of the square
    dx = np.abs(X - x0)
    dy = np.abs(Y - y0)
    max_dist = np.maximum(dx, dy)  # Distance from center to furthest edge axis
    distance = np.abs(max_dist - half_size)  # Distance to nearest square edge
    
    # Apply Gaussian
    gaussian = np.exp(-(distance ** 2) / (2 * sigma ** 2))
    
    return gaussian

def generate_generic_loop_gaussian(shape, parametric_curve, num_samples=1000, sigma=1.0):
    """
    Generate a Gaussian heatmap peaking along a parametric closed loop.
    
    Parameters:
    shape : tuple
        Output array shape (rows, cols).
    parametric_curve : function
        Function that takes `t` (array of parameters in [0, 1]) and returns 
        (x, y) coordinates of the closed loop.
    num_samples : int, optional
        Number of points to sample along the loop for distance computation.
    sigma : float, optional
        Controls the Gaussian decay from the loop.
    
    Returns:
    numpy.ndarray
        2D array with maximum values (1) along the loop.
    """
    rows, cols = shape
    
    # Generate grid coordinates
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    grid_points = np.column_stack((X.ravel(), Y.ravel()))  # Shape (N, 2)
    
    # Sample points along the parametric curve
    t = np.linspace(0, 1, num_samples)
    loop_points = parametric_curve(t)  # Shape (num_samples, 2)
    
    # Build KDTree for efficient nearest-neighbor search
    tree = KDTree(loop_points)
    
    # Compute distances from grid points to the loop
    distances, _ = tree.query(grid_points, k=1)
    distances = distances.reshape(rows, cols)
    
    # Compute Gaussian and normalize to max=1
    gaussian = np.exp(-(distances ** 2) / (2 * sigma ** 2))
    gaussian /= gaussian.max()
    
    return gaussian

def generate_edge_gaussian(seg_mask, sigma=1.0):
    """
    Generate a Gaussian heatmap peaking at the edges of a segmentation mask.
    
    Parameters:
    seg_mask : numpy.ndarray
        Binary 2D array (0s and 1s) representing the segmentation mask.
    sigma : float, optional
        Controls the Gaussian decay from the edges.
    
    Returns:
    numpy.ndarray
        2D Gaussian heatmap with maxima (1) at the mask edges.
    """
    # Ensure the mask is binary (0s and 1s)
    seg_mask = (seg_mask > 0).astype(np.uint8)
    
    # Step 1: Detect edges using erosion
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_mask = cv2.erode(seg_mask, kernel, iterations=1)
    edge_mask = seg_mask - eroded_mask  # Subtract eroded mask to get edges
    edge_mask = (edge_mask > 0).astype(np.uint8)  # Edges are 1, others 0
    
    # Step 2: Compute distance from every pixel to the nearest edge
    # Edge_mask is 1 at edges, 0 elsewhere. Compute distance to nearest 1 (edge)
    distance_map = ndimage.distance_transform_edt(1 - edge_mask)
    
    # Step 3: Apply Gaussian to the distance map
    gaussian = np.exp(-(distance_map ** 2) / (2 * sigma ** 2))
    
    # Normalize to ensure maximum value is 1
    gaussian /= gaussian.max()
    
    return gaussian

def circle_curve(t):
    theta = 2 * np.pi * t  # t ∈ [0, 1]
    R = 20  # Radius
    center_x, center_y = 50, 50  # Center of the grid (100x100)
    x = center_x + R * np.cos(theta)
    y = center_y + R * np.sin(theta)
    return np.column_stack((x, y))

def create_infinity_mask(shape, center=None, R=20):
    """
    Create a binary segmentation mask of a filled infinity symbol.
    
    Parameters:
    shape : tuple
        Output mask shape (rows, cols).
    center : tuple, optional
        Center (x, y) of the infinity symbol. Defaults to array center.
    R : float, optional
        Scaling factor for the infinity symbol size.
    
    Returns:
    numpy.ndarray
        Binary mask (0s and 1s) with the infinity symbol filled.
    """
    rows, cols = shape
    if center is None:
        center_x, center_y = cols // 2, rows // 2
    else:
        center_x, center_y = center
    
    # Generate points along the infinity symbol
    t = np.linspace(0, 2 * np.pi, 1000)  # Parameter t
    x = R * np.cos(t) + center_x  # x-coordinates
    y = R * np.sin(t) * np.cos(t) + center_y  # y-coordinates
    
    # Create an empty mask
    mask = np.zeros((rows, cols), dtype=np.uint8)
    
    # Convert points to integer coordinates
    points = np.column_stack((x.astype(int), y.astype(int)))
    
    # Fill the polygon defined by the infinity symbol
    cv2.fillPoly(mask, [points], color=1)
    
    return mask

# Generate a 100x100 Gaussian centered at (50, 50) with sigma=10
gaussian = generate_2d_gaussian((100, 100), (50, 50), sigma=10)

# Generate a 100x100 Gaussian peaking at a circle of radius 20
circular_gaussian = generate_circular_gaussian((100, 100), R=20, sigma=5)

# Generate a 100x100 Mexican hat profile with R=20, sigma_in=5, sigma_out=10
mexican_hat = generate_mexican_hat((100, 100), R=20, sigma_in=5, sigma_out=10)

# Generate a 100x100 Gaussian peaking at the edges of a square with half-size 20
square_edge_gaussian = generate_square_edge_gaussian((100, 100), half_size=20, sigma=5)

mask = np.zeros((100, 100), dtype=np.uint8)
mask[30:70, 30:70] = 1  # Binary square from (30,30) to (70,70)

infinity_mask = create_infinity_mask((100, 100), R=30)
# Generate the Gaussian heatmap
gaussian_general = generate_edge_gaussian(mask, sigma=5)
gaussian_infinity = generate_edge_gaussian(infinity_mask, sigma=5)

# To visualize using matplotlib
plt.figure()
plt.imshow(gaussian, cmap='hot')
plt.colorbar()

plt.figure()
plt.imshow(circular_gaussian, cmap='hot')
plt.colorbar()

plt.figure()
plt.imshow(mexican_hat, cmap='hot')
plt.colorbar()

plt.figure()
plt.imshow(square_edge_gaussian, cmap='hot')
plt.colorbar()

plt.figure()
plt.imshow(gaussian_general, cmap='hot')
plt.colorbar()

plt.figure(figsize=(14, 7), tight_layout=True)
plt.subplot(1, 2, 1)
plt.imshow(infinity_mask, cmap='gray')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gaussian_infinity, cmap='hot')
plt.axis('off')
plt.colorbar()

plt.show()

