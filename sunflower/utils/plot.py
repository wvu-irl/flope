import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from matplotlib.colors import hsv_to_rgb
import plotly.graph_objects as go

from sunflower.utils.conversion import qvec2rotmat
from sunflower.utils.mvg import project_3d_to_2d, make_homogeneous

def plot_bounding_boxes(image, bounding_boxes, color=(0,0,255)):
    """
    image (np.ndarray): Image
    bounding_boxes(np.ndarray or List[List[int]])
    """
    for bb in bounding_boxes:
        xmin, ymin, xmax, ymax = bb
        cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 10)
    return image
    
def plot_projected_box_to_image(image, cube_corners):
    """
    image: np.ndarray
        Image to plot cube into
    cube_corners: np.ndarray
        Cube corners of shape (8,2)
    """
    edges = [
        (0,1), (1,2), (2,3), (3,0),
        (4,5), (5,6), (6,7), (7,4),
        (4,0), (1,5), (2,6), (3,7)
    ]

    color = (255, 0, 0)  # Blue color
    thickness = 2        # Thickness of the line

    for i,j in edges:
        start_point = (int(cube_corners[i][0]), int(cube_corners[i][1]) )
        end_point = (int(cube_corners[j][0]), int(cube_corners[j][1]) )
        cv2.line(image, start_point, end_point, color, thickness)

 
def plot_axis(image, R, t, K, thickness=5):
    """Plots any pose as xyz axis, by projecting it to the image.
    Args:
        image (np.ndarray): Numpy Image, (H, W, 3)
        R: (3,3)
        t: (3,)
        K: (3,3)
        thickness (int): Thickness of Axis plot
    Returns:
        np.ndarray: Plotted Numpy Image, (H, W, 3)
    """
    points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    ])*0.05
    points2d = project_3d_to_2d(points, K, R, t)
    points2d = points2d.astype(np.int32)
   
    cv2.line(image, points2d[0], points2d[1], color=(0,0,255), thickness=thickness) # X-axis
    cv2.line(image, points2d[0], points2d[2], color=(0,255,0), thickness=thickness) # Y-axis
    cv2.line(image, points2d[0], points2d[3], color=(255,0,0), thickness=thickness) # Z-axis
        
    return image


def plot_axis_and_translate(image, R, t, K, bb, h, w, thickness=5):
    """
    Args:
        image (np.ndarray): Numpy Image, (H, W, 3)
        R: (3,3)
        t: (3,)
        K: (3,3)
        b: bounding box
        h: image height
        w: image width
        thickness (int): Thickness of Axis plot
    Returns:
        np.ndarray: Plotted Numpy Image, (H, W, 3)
    """
    xmin, ymin, xmax, ymax = bb
    # x_offset = xmin+(xmax-xmin)/2 - 2016
    # y_offset = ymin+(ymax-ymin)/2 - 1512
    
    x_offset = xmin+(xmax-xmin)/2 - w/2
    y_offset = ymin+(ymax-ymin)/2 - h/2
    
    points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    ])*0.05
    
    
    points2d = project_3d_to_2d(points, K, R, t)
   
    points2d[:,0]  += x_offset
    points2d[:,1]  += y_offset
    
    # print(points2d.shape)
    # exit()
    points2d = points2d.astype(np.int32)
   
    cv2.line(image, points2d[0], points2d[1], color=(0,0,255), thickness=thickness) # X-axis
    cv2.line(image, points2d[0], points2d[2], color=(0,255,0), thickness=thickness) # Y-axis
    cv2.line(image, points2d[0], points2d[3], color=(255,0,0), thickness=thickness) # Z-axis
        
    return image


def plot_3D_poses(ax, trans, quat):
    rotmat = R.from_quat(quat).as_matrix()
    obj_dirn = np.array([0,0,1])
    rotated_obj = rotmat@obj_dirn
    
    ax.scatter(trans[:,0], trans[:,1], trans[:,2], cmap='viridis', s=50, marker='o')
    ax.quiver(trans[:,0], trans[:,1], trans[:,2], rotated_obj[:,0], rotated_obj[:,1], rotated_obj[:,2], length=0.1, normalize=True)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    
def plotly_flower_poses(trans, rotmat):
    obj_dirn = np.array([0,0,-1])
    rotated_obj = rotmat@obj_dirn
    
    flower_cones = go.Cone(
        x=trans[:,0],
        y=trans[:,1],
        z=trans[:,2],
        u=rotated_obj[:,0],
        v=rotated_obj[:,1],
        w=rotated_obj[:,2],
        sizemode="absolute",
        sizeref=1,
        anchor="tip",
        colorscale=[[0, 'blue'], [1, 'blue']],  # Uniform blue color
        cmin=0,
        cmax=1,
        showscale=False  # Hide color scale
    )
    
    return [flower_cones]
    
    
def generate_rainbow_colors(N):
    """
    Generate N RGB values transitioning smoothly from red to violet.
    """
    # Linearly interpolate hue values from red (0) to violet (0.83 in HSV)
    hues = np.linspace(0, 0.83, N)
    
    # Create HSV values (full saturation and value for vivid colors)
    hsv_colors = np.array([[hue, 1, 1] for hue in hues])
    
    # Convert HSV to RGB
    rgb_colors = hsv_to_rgb(hsv_colors)
    
    return rgb_colors


def toy_intrinsics():
    return {'w':6, 'h':4, 'fx':10, 'fy':10, 'cx': 3, 'cy': 2}

def get_identity_cam(scale=1, cam_orientation=(1,1,1), params=toy_intrinsics()):
    """
    Gets an identity camera plot

    Parameters
    ----------
    scale: int
        Scale of the camera plot

    Returns
    -------
    cam_plot: np.ndarray
        (11, 3) Sequence of points to make camera plot
    axis: np.ndarray
        (4,3) Axis points of camera
    params: dict
        Camera intirisic parameters
    """
    f = params['fx']
    w = params['w']/2
    h = params['h']/2

    cam_orientation = np.array(cam_orientation)
    unit_cam = np.array([
        [0,0,0],
        [w,-h,f],
        [w,h,f],
        [-w,h,f],
        [-w,-h,f],
        [0,-2*h,f]
    ])
    unit_cam *= cam_orientation

    seq = np.array([3,4,1,2,0,1,5,4,0,3,2])
    draw_cam = unit_cam[seq]

    axis = np.array([
        [0,0,0],
        [3,0,0],
        [0,3,0],
        [0,0,3]
    ])

    return draw_cam*scale, axis*scale


def get_cam_plot(qvec, tvec, unit_cam, axis):
    """
    Gets camera plot, pose defined by given quaternion and translation

    Parameters
    ----------
    qvec: np.ndarray
        (4,) Quaternion of Camera Pose
    tvec: np.ndarray
        (3,) Translation of Camera Pose
    unit_cam: np.ndarray
        (11,3) Ideal camera sequence

    Returns
    -------
    cam: np.ndarray
        (11,3) Sequence of corrdinates to draw given camera pose
    axis: np.ndarray
        (4,3) Axis points of camera
    """
    R = qvec2rotmat(qvec)
    rotated_cam = (R@unit_cam.T + tvec.reshape(3,1)).T
    rotated_axis = (R@axis.T + tvec.reshape(3,1)).T
    return rotated_cam, rotated_axis


def plotly_single_cam(cam, cam_name=None, color='black'):
    """
    Get Plotly Camera plot for one camera

    Parameters
    ----------
    cam: np.ndarray
        (11,3) Sequence of corrdinates to draw given camera pose

    Returns
    -------
    cam_plot: go.Scatt3d
        Camera Plot
    """
    cam_plot_complete = []

    cam_plot = go.Scatter3d(
        x=cam[:,0],
        y=cam[:,1],
        z=cam[:,2],
        mode='lines',
        marker=dict(
            size=2,
            opacity=1
        ),
        line=dict(
            width=4,
            color=color,
        ),
        showlegend=False,
    )
    cam_plot_complete.append(cam_plot)

    if cam_name is not None:
        cam_name_plot = go.Scatter3d(
            x=cam[4,0][None],
            y=cam[4,1][None],
            z=cam[4,2][None],
            mode='text',
            marker=dict(
                size=2,
                opacity=1
            ),
            showlegend=False,
            text=cam_name
        )
        cam_plot_complete.append(cam_name_plot)

    return cam_plot_complete


def plotly_cam_poses(poses: np.ndarray, cam_orientation=(1,1,1), scale=0.02, 
                     params=toy_intrinsics(),
                     alpha=1, color=None, linestyle='-' , linewidth=1,
                     plot_axis=False, axis_alpha=1, plot_cam_count=True, cam_names=None):
    """
    Plot Camera Poses and Camera Axis (optional)

    Parameters
    ----------
    poses: np.ndarray
        (N,7) Camera Poses

    Returns
    -------
    all_cams_plot: go.Scatter3d
        List of Plotly graph object Scatter Plot
    """
    
    identity_cam, identity_axis = get_identity_cam(scale, cam_orientation, params)
    all_cams_plot = []
    for i in range(poses.shape[0]):
        pos = poses[i]
        count = i

        # cam color
        if color is None:
            if isinstance(color, list):
                col = color[i]
            else:
                col = color
        else:
            col = 'black'

        # cam name
        if plot_cam_count:
            cname = str(count)
        elif cam_names != None:
            cname = cam_names[i]
        else:
            cname = None

        cam, axis = get_cam_plot(pos[:4], pos[4:], identity_cam, identity_axis)
        all_cams_plot += plotly_single_cam(cam, cam_name=cname, color=col)

    #TODO axis plot

    return all_cams_plot


def plotly_axis():
    scale = 0.1
    
    x_plot = go.Scatter3d(
        x=[0,scale],
        y=[0,0],
        z=[0,0],
        mode='lines',
        marker=dict(
            size=2,
            opacity=1
        ),
        line=dict(
            width=7,
            color='red'
        ),
        showlegend=False,
    )
    y_plot = go.Scatter3d(
        x=[0,0],
        y=[0,scale],
        z=[0,0],
        mode='lines',
        marker=dict(
            size=2,
            opacity=1
        ),
        line=dict(
            width=7,
            color='green'
        ),
        showlegend=False,
    )
    z_plot = go.Scatter3d(
        x=[0,0],
        y=[0,0],
        z=[0,scale],
        mode='lines',
        marker=dict(
            size=2,
            opacity=1
        ),
        line=dict(
            width=7,
            color='blue'
        ),
        showlegend=False,
    )
    return [x_plot, y_plot, z_plot]


def write_text_bottom(text, image):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 3
    font_thickness = 5
    color = (0, 0, 255)  # White text in BGR

    # Get the image dimensions
    image_height, image_width = image.shape[:2]

    # Get the text size
    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

    # Calculate the text's x and y position
    x = (image_width - text_size[0]) // 2
    y = image_height - 10  # 10 pixels above the bottom edge

    # Add the text to the image
    cv2.putText(image, text, (x, y), font, font_scale, color, font_thickness)


def write_text(text, image, xy):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_thickness = 5
    color = (0, 0, 255)  # White text in BGR

    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]


    # Add the text to the image
    cv2.putText(image, text, xy, font, font_scale, color, font_thickness)


def apply_depth_colormap(depth_clone):
    depth_clone = (depth_clone-depth_clone.min())/(depth_clone.max()-depth_clone.min())
    depth_clone = depth_clone*255.0
    depth_clone = depth_clone.astype(np.uint8)
    depth_clone = cv2.applyColorMap(depth_clone, cv2.COLORMAP_JET)
    return depth_clone


def plot_flower_poses_on_image(image, Rts, K, plot_count=True, plot_distance=True):
    """
    Flower poses should be in camera coordinate system

    REMEMBER: these Rts are not the camera poses or camera extrinsics.
    these are simple rotation and translation of the flower poses w.r.t the camera
    so, simple transform identity axis by these Rts to get the flower axises in 
    the camera coordinate system.
    then apply K of the camera to project it to the image plane

    Args:
        image: Numpy image
        Rts: Flower poses in cam coordinate (Nx4x4)
    """

    identity_axis = make_homogeneous(np.vstack((np.zeros(3), np.eye(3)*0.02)))
    K_4by4 = np.eye(4)
    K_4by4[:3,:3] = K

    for count,Rt in enumerate(Rts):
        trasformed_axis = (Rt@identity_axis.T).T
        projected_axis = (K_4by4@trasformed_axis.T).T

        projected_axis = projected_axis[:,:3]
        projected_axis /= projected_axis[:,2].reshape(-1,1)
        projected_axis = projected_axis[:,:2]

        projected_axis = projected_axis.astype(np.int16)


        #! Plot
        for i,axis in enumerate(projected_axis[1:]):
            color = np.zeros(3)
            color[2-i] = 255
            cv2.line(
                image,
                projected_axis[0], axis,
                color=color, thickness=10
            )

        if plot_distance and plot_count:
            dist_to_flower = np.linalg.norm(Rt[:3,3])
            img_label = f"{count}:{int(dist_to_flower*100)}cm"
        elif plot_distance and not plot_count:
            dist_to_flower = np.linalg.norm(Rt[:3,3])
            img_label = f"{int(dist_to_flower*100)}cm"
        elif not plot_distance and plot_count:
            img_label = f"{count}"
        else:
            img_label = None
           
        if img_label is not None:
            cv2.putText(
                image, 
                img_label, 
                org=projected_axis[0], 
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0,0,255),
                thickness=5
            )