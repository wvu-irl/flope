import cv2
import numpy as np

# Set parameters
num_markers_x = 4     # Number of markers in each row
num_markers_y = 6     # Number of markers in each column
marker_size = 300     # Size of each marker in pixels
margin = 100           # Margin between markers in pixels

# Define the dictionary of ArUco markers
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)

# Create a blank image to hold the matrix
grid_width = num_markers_x * (marker_size + margin) + margin
grid_height = num_markers_y * (marker_size + margin) + margin
grid_image = np.ones((grid_height, grid_width), dtype=np.uint8) * 255  # White background

# Generate and place each marker in the grid
for i in range(num_markers_y):
    for j in range(num_markers_x):
        # Generate the marker
        marker_id = i * num_markers_x + j  # Unique ID for each marker
        marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
        
        # Compute the position in the grid
        x = j * (marker_size + margin)
        y = i * (marker_size + margin)
        
        # Place the marker in the grid
        grid_image[y+margin:y+marker_size+margin, x+margin:x+marker_size+margin] = marker_img

radius = 10
color = (0,0,0)
thickness = -1

jump = marker_size+margin
for i in range(num_markers_y-1):
    for j in range(num_markers_x-1):
        x = int(margin+marker_size+margin/2 + i*jump)
        y = int(margin+marker_size+margin/2 + j*jump)
        image = cv2.circle(grid_image, (y,x), radius, color, thickness) 

# Save or display the grid image
cv2.imwrite("aruco_grid.png", grid_image)
cv2.imshow("ArUco Marker Grid", grid_image)
cv2.waitKey(0)
cv2.destroyAllWindows()