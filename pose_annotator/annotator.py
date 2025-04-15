import numpy as np
from scipy.spatial.transform import Rotation as scipyR
from dash import Dash, dcc, html, Output, Input, State, ctx, no_update
import plotly.graph_objects as go

from annotator_utils import plotly_poses, plotly_flower_model


class PlotData:
    def __init__(self, flower_poses_file):
        self.delta_trans = 0.01
        self.delta_rot = 10
        # self.poses = read_flower_poses_data(flower_poses_file)
        self.poses = np.load('gt_poses.npy')
        self.N = self.poses.shape[0]
        print(f"{self.N} poses loaded.")
        self.selected_pose = 0
        self.ids = np.arange(self.N)
        self.mask = np.array(self.N*[True])
        
        self.camera = dict(
            eye=dict(x=0, y=1.25, z=-1.0),  # Initial camera position
            up=dict(x=0, y=0, z=-1),
        )
        
        self.clicked_point = [0,0,0]
        
    def get_rotmat(self, axis, dirn):
        delta_rot = dirn*self.delta_rot
        if axis=='x':
            rotmat = scipyR.from_euler('xyz', [delta_rot, 0, 0],degrees=True).as_matrix()
        elif axis=='y':
            rotmat = scipyR.from_euler('xyz', [0, delta_rot, 0],degrees=True).as_matrix()
        elif axis=='z':
            rotmat = scipyR.from_euler('xyz', [0, 0, delta_rot],degrees=True).as_matrix()
            
        # make 3x3 -> 4x4
        rotmat = np.hstack((rotmat, np.zeros((3,1)) ))
        rotmat = np.vstack((rotmat, np.array([0,0,0,1])))
        
        return rotmat

    def save_poses(self):
        np.save(filename:='gt_poses.npy', self.poses[self.mask])
        print(f"Poses saved to: {filename}")
        return filename

    def add_new_pose(self):
        identity_pose = np.eye(4)
        identity_pose[0,3] = self.clicked_point[0]
        identity_pose[1,3] = self.clicked_point[1]
        identity_pose[2,3] = self.clicked_point[2]
        identity_pose = identity_pose[None]
        self.poses = np.concatenate((self.poses, identity_pose))
        self.N = self.poses.shape[0]
        self.ids = np.arange(self.N)
        self.mask = np.concatenate((self.mask, np.array([True])))
        self.selected_pose = self.ids[-1]

    def remove_pose(self, num):
        self.mask[num] = False
            
        
app = Dash('Manual Data Generator')
data = PlotData(flower_poses_file='data/posenet_out.pkl')

@app.callback(
    Output("poses_plot", "figure", allow_duplicate=True),
    Output("clicked_point", "children", allow_duplicate=True),
    [Input("poses_plot", "relayoutData")],
    prevent_initial_call=True
)
def display_camera_orientation(relayout_data):
    if relayout_data and "scene.camera" in relayout_data:
        camera = relayout_data["scene.camera"]
        data.camera = camera
    return no_update, f"  Point: ({data.clicked_point[0]:.2f}, {data.clicked_point[1]:.2f}, {data.clicked_point[2]:.2f})"  # Prevent the plot from being re-rendered


@app.callback(
    Output("poses_plot", "figure", allow_duplicate=True),
    [Input("poses_plot", "clickData")],
    prevent_initial_call=True
)
def get_clicked_coordinates(click_data):
    if click_data and "points" in click_data:
        # Extract the first clicked point's coordinates
        point = click_data["points"][0]
        x, y, z = point["x"], point["y"], point["z"]
        # print(f"Clicked Point Coordinates: x={x:.2f}, y={y:.2f}, z={z:.2f}")
        data.clicked_point = (x,y,z)
    return no_update  # Prevent the figure from updating


@app.callback(
    Output("remove_pose_log", "children"),
    Output("poses_plot", "figure", allow_duplicate=True),
    Output("select_pose", "value"),
    Output("select_pose", "options", allow_duplicate=True),

    Input("remove_pose", "n_clicks"),

    State("select_pose", "value"),

    prevent_initial_call=True
)
def remove_pose(n_clicks, pose_number):
    if n_clicks>0:
        data.remove_pose(pose_number)
        pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()
        fig = go.Figure(data=pose_plot,layout=dict(scene=dict(dragmode="orbit",camera=data.camera)))
        return f"Pose {pose_number} removed", fig, 0, data.ids[data.mask]
    else:
        pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()
        fig = go.Figure(data=pose_plot,layout=dict(scene=dict(dragmode="orbit",camera=data.camera)))
        return "", fig, pose_number, data.ids[data.mask]


@app.callback(
    Output("add_pose_log", "children"),
    Output("select_pose", "options"),
    Output("select_pose", "value", allow_duplicate=True),
    Output("poses_plot", "figure", allow_duplicate=True),

    Input("add_pose", "n_clicks"),

    State("select_pose", "options"),

    prevent_initial_call=True
)
def add_pose(n_clicks, current_ids):
    if n_clicks>0:
        data.add_new_pose()
        pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()
        fig = go.Figure(data=pose_plot,layout=dict(scene=dict(dragmode="orbit",camera=data.camera)))
        return "Pose added", data.ids[data.mask], data.selected_pose, fig
    else:
        pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()
        fig = go.Figure(data=pose_plot,layout=dict(scene=dict(dragmode="orbit",camera=data.camera)))
        return "", current_ids, data.selected_pose, fig


@app.callback(
    Output("delT_vis", "children"),
    Input("pos_delT", "n_clicks"),
    Input("neg_delT", "n_clicks")
)
def update_delT(pos_delT_clicks, neg_delT_clicks):
    button_id = ctx.triggered_id if not None else 'No clicks yet'

    if button_id == 'pos_delT':
        data.delta_trans *= 10
    elif button_id == 'neg_delT':
        data.delta_trans /= 10

    return f"ΔT: {data.delta_trans}"


@app.callback(
    Output("delR_vis", "children"),
    Input("pos_delR", "n_clicks"),
    Input("neg_delR", "n_clicks")
)
def update_delR(pos_delR_clicks, neg_delR_clicks):
    button_id = ctx.triggered_id if not None else 'No clicks yet'

    if button_id == 'pos_delR':
        data.delta_rot *= 10
    elif button_id == 'neg_delR':
        data.delta_rot /= 10

    return f"ΔR: {data.delta_rot}"


@app.callback(
    Output("save_pose_log", "children"),
    Input("save_pose", "n_clicks")
)
def save_pose(save_pose_clicks):
    if save_pose_clicks>0:
        filename = data.save_poses()
        return f"Poses saved to {filename}"
    else:
        return ""


@app.callback(
    Output("poses_plot", "figure", allow_duplicate=True),
    
    Input("pos_x", "n_clicks"),
    Input("neg_x", "n_clicks"),
    Input("pos_y", "n_clicks"),
    Input("neg_y", "n_clicks"),
    Input("pos_z", "n_clicks"),
    Input("neg_z", "n_clicks"),
    
    Input("pos_rx", "n_clicks"),
    Input("neg_rx", "n_clicks"),
    Input("pos_ry", "n_clicks"),
    Input("neg_ry", "n_clicks"),
    Input("pos_rz", "n_clicks"),
    Input("neg_rz", "n_clicks"),
    
    State("poses_plot", "figure"),
    State("select_pose", "value"),

    prevent_initial_call=True
)
def update_fig(
    pos_x_clicks, neg_x_clicks, pos_y_clicks, neg_y_clicks, pos_z_clicks, neg_z_clicks, 
    pos_rx_clicks, neg_rx_clicks, pos_ry_clicks, neg_ry_clicks, pos_rz_clicks, neg_rz_clicks,
    current_figure, selected_pose):
    # figure out which element has triggered this callback
    button_id = ctx.triggered_id if not None else 'No clicks yet'
    data.selected_pose = selected_pose    

    if button_id=='pos_x':
        if pos_x_clicks > 0:
            data.poses[data.selected_pose][0,3] += data.delta_trans

    elif button_id=='neg_x':
        if neg_x_clicks > 0:
            data.poses[data.selected_pose][0,3] -= data.delta_trans

    elif button_id=='pos_y':
        if pos_y_clicks > 0:
            data.poses[data.selected_pose][1,3] += data.delta_trans

    elif button_id=='neg_y':
        if neg_y_clicks > 0:
            data.poses[data.selected_pose][1,3] -= data.delta_trans
            
    elif button_id=='pos_z':
        if pos_z_clicks > 0:
            data.poses[data.selected_pose][2,3] += data.delta_trans

    elif button_id=='neg_z':
        if neg_z_clicks > 0:
            data.poses[data.selected_pose][2,3] -= data.delta_trans 
            
    
    elif button_id=='pos_rx':
        if pos_rx_clicks > 0:
            rotmat = data.get_rotmat('x', 1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat

    elif button_id=='neg_rx':
        if neg_rx_clicks > 0:
            rotmat = data.get_rotmat('x', -1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat
            
    elif button_id=='pos_ry':
        if pos_ry_clicks > 0:
            rotmat = data.get_rotmat('y', 1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat

    elif button_id=='neg_ry':
        if neg_ry_clicks > 0:
            rotmat = data.get_rotmat('y', -1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat
            
    elif button_id=='pos_rz':
        if pos_rz_clicks > 0:
            rotmat = data.get_rotmat('z', 1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat

    elif button_id=='neg_rz':
        if neg_rz_clicks > 0:
            rotmat = data.get_rotmat('z', -1)
            data.poses[data.selected_pose] = data.poses[data.selected_pose]@rotmat

    pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()
    return go.Figure(data=pose_plot,layout=dict(scene=dict(dragmode="orbit",camera=data.camera)))

#! Get Plot Data
pose_plot = plotly_poses(data.poses[data.mask], data.ids[data.mask])+plotly_flower_model()

#! App Layout
app.layout = html.Div([
    html.H1('FloPE Pose Annotator Tool'),
    html.Div([
        html.Plaintext('Select Pose:'),
        dcc.Dropdown(data.ids[data.mask], 0, id='select_pose', style={"width": "100px"}),
        html.Button("Save Poses", id="save_pose", n_clicks=0),
        html.Plaintext("", id='save_pose_log'),
        html.Button("Add Pose", id="add_pose", n_clicks=0),
        html.Plaintext("", id='add_pose_log'),
        html.Button("Remove Pose", id="remove_pose", n_clicks=0),
        html.Plaintext("", id='remove_pose_log'),
        html.Plaintext("", id='clicked_point'),
    ], style={"display": "flex", "gap": "5px"}),
    dcc.Graph(
        id="poses_plot",
        figure=go.Figure(data=pose_plot, layout=dict(scene=dict(dragmode="orbit",camera=data.camera))),
        style={"height": "80vh"}
    ),
    html.Div([
        #! Translation Buttons
        html.Div([
            html.Button("+X", id="pos_x", n_clicks=0),
            html.Button("-X", id="neg_x", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        html.Div([
            html.Button("+Y", id="pos_y", n_clicks=0),
            html.Button("-Y", id="neg_y", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        html.Div([
            html.Button("+Z", id="pos_z", n_clicks=0),
            html.Button("-Z", id="neg_z", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
       
        #! Rotation Buttons 
        html.Div([
            html.Button("+Rx", id="pos_rx", n_clicks=0),
            html.Button("-Rx", id="neg_rx", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        html.Div([
            html.Button("+Ry", id="pos_ry", n_clicks=0),
            html.Button("-Ry", id="neg_ry", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        html.Div([
            html.Button("+Rz", id="pos_rz", n_clicks=0),
            html.Button("-Rz", id="neg_rz", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),

        #! Change Resolution Buttons
        html.Div([
            html.Button("+ΔT", id="pos_delT", n_clicks=0),
            html.Button("-ΔT", id="neg_delT", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        html.Div([
            html.Button("+ΔR", id="pos_delR", n_clicks=0),
            html.Button("-ΔR", id="neg_delR", n_clicks=0),
        ], style={"display": "flex", "flex-direction": "column", "width": "50px"}),
        
        
                
    ], style={"display": "flex"}),
    html.Div([
        html.Plaintext(f"ΔT: {data.delta_trans}", id='delT_vis'),
        html.Plaintext(f"ΔR: {data.delta_rot}", id='delR_vis')
    ], id="delta_vis")
])

app.run_server(debug=False)