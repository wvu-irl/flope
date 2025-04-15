import pickle
from scipy.spatial.transform import Rotation as scipyR
import numpy as np
import plyfile
import json
import plotly.graph_objects as go


def make_homogeneous(points):
    """Adds a ones column at the end
    
    Args:
        points: N points. Can be (N,2) or (N,3)
        
    Returns:
        Homogeneous points. Can be (N,3) or (N,4)
    """
    N = points.shape[0]
    ones_column = np.ones(N).reshape(-1,1)
    homogeneous = np.hstack((points, ones_column))
    return homogeneous


def get_Rt(R, t):
    """Given R and t, get Rt (a Nx4x4 mat)
    """
    Rt = np.zeros((R.shape[0], 4, 4))
    Rt[:, :3, :3] = R
    Rt[:, :3, 3] = t
    Rt[:, 3, 3] = 1
    return Rt


def read_splats_ply(splats_path):
    with open(splats_path, "rb") as f:
        ply_data = plyfile.PlyData.read(f)
    
    vertex_data = ply_data["vertex"]

    x = np.array(vertex_data["x"])
    y = np.array(vertex_data["y"])
    z = np.array(vertex_data["z"])

    r = np.array(vertex_data["f_dc_0"])
    g = np.array(vertex_data["f_dc_1"])
    b = np.array(vertex_data["f_dc_2"])

    points = np.array([x,y,z]).T
    colors = np.array([r,g,b]).T

    colors = (colors-colors.min())/(colors.max()-colors.min())
    
    return points, colors


def read_dataparser_transforms(filepath):
    # Read transforms
    with open(filepath, 'r') as f:
        splat_tf = json.load(f)
        
    splat_Rt = np.array(splat_tf['transform']) # Gives 3x4 mat
    splat_Rt = np.vstack((splat_Rt, np.array([0,0,0,1]))) # Get 4x4 mat
    splat_scale = splat_tf['scale']
    # Invert the transformation
    splat_Rt = np.linalg.inv(splat_Rt)
    splat_scale = 1/splat_scale
    
    return splat_Rt, splat_scale


def scale_and_rotate_points(points, Rt, scale):
    # Apply scaling
    points *= scale
    # Apply Rt
    points = np.hstack((points, np.ones(points.shape[0]).reshape(-1,1))) # Convert to homogeneous coordinate
    points = (Rt@points.T).T # Do rotation
    points /= points[:,-1].reshape(-1,1) # Convert back to normal from homogeneous
    points = points[:,:-1] # Get rid of last homogeneous dim (all ones how after above step)
    return points


def plotly_xyzrgb(points, colors):
    point_cloud = go.Scatter3d(
        x=points[:,0], 
        y=points[:,1], 
        z=points[:,2], 
        mode='markers',
        marker=dict(
            size=2,
            color=colors
        ),
        name='Point Cloud',         # 👈 Legend label
        showlegend=False             # 👈 Show in legend
    )
    return [point_cloud]

def get_splats_plot(splat_points, splat_color):
    point_cloud = go.Scatter3d(
        x=splat_points[:,0], 
        y=splat_points[:,1], 
        z=splat_points[:,2], 
        mode='markers',
        marker=dict(
            size=2,
            color=splat_color
        )
    )
    return [point_cloud]
    
def plotly_poses(Rctc, names, scale=0.1):
    """ Plot 3D poses in plotly
    
    Args:
        Rctc: Nx4x4 Pose Matrix
    Returns:
        List of Plotly graph object for pose plot
    """
    identity_pose = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ]).astype(np.float32)
    identity_pose *= scale
    identity_pose = make_homogeneous(identity_pose)
        
    poses_plot = [] 
    for name, rctc in zip(names, Rctc):
        transformed_pose = (rctc@identity_pose.T).T
        transformed_pose = transformed_pose[:,:-1]
        poses_plot += plotly_pose(transformed_pose, name)
    return poses_plot
  

def plotly_pose(identity_pose, name):
    x_plot = go.Scatter3d(
        x=identity_pose[[0,1],0],
        y=identity_pose[[0,1],1],
        z=identity_pose[[0,1],2],
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
        x=identity_pose[[0,2],0],
        y=identity_pose[[0,2],1],
        z=identity_pose[[0,2],2],
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
        x=identity_pose[[0,3],0],
        y=identity_pose[[0,3],1],
        z=identity_pose[[0,3],2],
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
   
    if name is not None: 
        name_plot = go.Scatter3d(
                x=identity_pose[0,0][None],
                y=identity_pose[0,1][None],
                z=identity_pose[0,2][None],
                mode='text',
                marker=dict(
                    size=10,
                    opacity=1
                ),
                showlegend=False,
                text=str(name)
            )
    
        return [x_plot, y_plot, z_plot, name_plot]
    else:
        return [x_plot, y_plot, z_plot]
        

def plot_3d(all_plots):
    fig = go.Figure(data=all_plots)
    fig.update_layout(
        title='3D Plot', 
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
    )
    fig.show()
    # fig.write_html("new_plot.html")
    # print(f"3D plot written to: new_plot.html")
    
   
 
def read_flower_poses_data(filename, det_threshold=30):
    with open(filename, 'rb') as fp:
        data = pickle.load(fp)
        print("Posenet Out loaded")
    
    trans, quat, scores = data['trans'], data['quat'], data['score']
    mask = scores>det_threshold
    trans = trans[mask]
    quat = quat[mask]
    rotmat = scipyR.from_quat(quat).as_matrix()
    Rt = get_Rt(rotmat, trans)
    return Rt


def get_flower_model():
    #! Read splats
    points, colors = read_splats_ply('data/plant_3dgs_model_cropped.ply')

    #! Read transforms
    with open('data/dataparser_transforms.json', 'r') as f:
        splat_tf = json.load(f)
        
    splat_Rt = np.array(splat_tf['transform']) # Gives 3x4 mat
    splat_Rt = np.vstack((splat_Rt, np.array([0,0,0,1]))) # Get 4x4 mat
    splat_scale = splat_tf['scale']

    #! Invert the transformation
    splat_Rt = np.linalg.inv(splat_Rt)
    splat_scale = 1/splat_scale

    #! Apply scaling
    print("Splats scaling is:", splat_scale)
    points *= splat_scale

    #! Apply Rt
    points = np.hstack((points, np.ones(points.shape[0]).reshape(-1,1))) # Convert to homogeneous coordinate
    points = (splat_Rt@points.T).T # Do rotation
    points /= points[:,-1].reshape(-1,1) # Convert back to normal from homogeneous
    points = points[:,:-1] # Get rid of last homogeneous dim (all ones how after above step)


    #! Get colors in appropriate format for plotly
    colors = (colors*255).astype(np.uint8)
    # rgb = []
    # for color in colors:
    #     rgb.append(f"rgb({color[0]},{color[1]},{color[2]})")

    return points, colors, splat_scale


def plotly_flower_model():
    points, colors, splat_scale = get_flower_model()
    return plotly_xyzrgb(points, colors)