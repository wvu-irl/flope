import numpy as np

def get_pixel6a_intrinsincs():
    return {
        'w': 1920,
        'h': 1080,
        'fx': 1751.276576,
        'fy': 1756.389162,
        'cx': 957.984186,
        'cy': 529.393387,
        'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
    }


def get_pixel6a_cam_matrix():
    cam_intrinsics = get_pixel6a_intrinsincs()
    return np.array([
        [cam_intrinsics['fx'], 0, cam_intrinsics['cx']],
        [0, cam_intrinsics['fy'], cam_intrinsics['cy']],
        [0, 0, 1]
    ])
    
    
def get_realsense_435_cam_matrix():
    return np.array([
        [1.361945190429687500e+03, 0.000000000000000000e+00, 9.635921630859375000e+02],
        [0.000000000000000000e+00, 1.361130371093750000e+03, 5.339596557617187500e+02],
        [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]
    ])

    
def get_obj_positions_on_aruco_grid(): 
    return [
        (56.0, -56.0, 17.0),
        (120.0, -184.0, 17.0), 
        (-8.0, -184.0, 17.0), 
        (-8.0, 72.0, 17.0),
        (120.0, 72.0, 17.0)
    ]