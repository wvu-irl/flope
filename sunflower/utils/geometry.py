import numpy as np
from shapely.geometry import Polygon


def shrink_quadrilateral(quad, width, height):
    """
    Shrink a quadrilateral inward so that rectangles of specified width and height
    can fit completely within it.
    
    Args:
        quad: List of 4 points (P1, P2, P3, P4) defining the quadrilateral.
        width: Width of the rectangle.
        height: Height of the rectangle.
    
    Returns:
        A shrunk quadrilateral as a list of 4 points.
    """
    
    # Create the original quadrilateral as a Polygon
    poly = Polygon(quad)
    
    # Offset inward by half the width and height
    # offset_dist = min(width, height) / 2
    # offset_dist = np.sqrt((width/2)**2+(height/2)**2)
    offset_dist = min(width/2, height/2)
    shrunk_poly = poly.buffer(-offset_dist)
    
    if shrunk_poly.is_empty:
        print('shrunk polygon is empty')
        return None  # No valid area for the rectangle
    
    # Extract the shrunk polygon's vertices
    shrunk_coords = list(shrunk_poly.exterior.coords)
    return np.array(shrunk_coords[:-1], dtype=np.float32)  # Remove duplicate last point


def sample_point_in_polygon(polygon):
    """
    Sample a random point inside a polygon using barycentric sampling.
    
    Args:
        polygon: List of 4 points defining the polygon (clockwise or counterclockwise).
    
    Returns:
        A sampled point (x, y) inside the polygon.
    """
    # Convert polygon to numpy array
    poly = np.array(polygon, dtype=np.float32)
    
    # Use barycentric coordinates to sample
    while True:
        weights = np.random.rand(4)
        weights /= np.sum(weights)
        point = np.dot(weights, poly)
        return tuple(point)
    
  
def get_rect_from_center(center, width, height):
    # Calculate rectangle corners based on the center
    cx, cy = center
    rect = [
        (cx - width / 2, cy - height / 2),  # Top-left
        (cx + width / 2, cy - height / 2),  # Top-right
        (cx + width / 2, cy + height / 2),  # Bottom-right
        (cx - width / 2, cy + height / 2)   # Bottom-left
    ] 
    return np.array(rect)

def sample_rectangle(corners, height, width):
    shrinked_corners = shrink_quadrilateral(corners, width, height)
    
    if shrinked_corners is None:
        return None, None
    
    center = sample_point_in_polygon(shrinked_corners)
    new_rect_corners = get_rect_from_center(center, width, height)
    return new_rect_corners, shrinked_corners