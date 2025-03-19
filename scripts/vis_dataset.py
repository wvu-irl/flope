#! Plotly code here
import os
from pathlib import Path
import random
import json
import numpy as np
import cv2
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as scipyR

from sunflower.utils.plot import plot_bounding_boxes
from sunflower.utils.mvg import get_points3d
from sunflower.utils.image_manipulation import get_depth_value
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w
from sunflower.utils.plot import generate_rainbow_colors, write_text_bottom, write_text
from sunflower.utils.plot import plotly_cam_poses, plotly_axis
from sunflower.utils.io import read_splats_ply


class VisDataset():
    def __init__(self, path:str, N:int):
        self.data_path = Path(path)
        self.N = N
        self.K, self.H, self.W = read_intrinsics_yaml_to_K_h_w(
            self.data_path/'intrinsics.yaml'
        )
        self.splat_transforms_pth = '/home/rashik_shrestha/outputs/plantscan_pixel_1230_sai/splatfacto/2024-12-30_190922/dataparser_transforms.json'

        self.color = generate_rainbow_colors(self.N)
        plotly_color = []
        for col in self.color:
            rgb = (col*255).astype(np.uint8)
            plotly_color.append(f"rgb({rgb[0]},{rgb[1]},{rgb[2]})")
            
        self.plotly_color = plotly_color

 
    def get_sample_names(self):
        """Get N sample names to visualize"""
        rgb_names = os.listdir(self.data_path/'rgb')
        sample_names = random.sample(rgb_names, self.N)
        self.sample_names = [name[:-4] for name in sample_names]

    
    def get_cam_poses(self):
        poses = []
        for sample in self.sample_names:
            poses.append(np.loadtxt(self.data_path/f"pose/{sample}.txt"))
        poses = np.array(poses) 

        # convert to quaternions and translations
        ts = poses[:,9:]
        Rs = poses[:,:9].reshape(-1,3,3)
        qs = scipyR.from_matrix(Rs).as_quat()
        qts = np.hstack((qs, ts))
        
        self.Rc = Rs
        self.tc = ts
        self.qc = qs
        self.qctc = qts
        
        
    def get_rgb_depth(self):
        rgb = []
        depth = []
        mask = []
        for sample in self.sample_names:
            rgb.append(cv2.imread(self.data_path/f"rgb/{sample}.jpg"))
            # depth.append(cv2.imread(self.data_path/f"depth/{sample}.png", cv2.IMREAD_UNCHANGED))
            depth.append(np.load(self.data_path/f"depth/{sample}.npy"))
            mask.append(cv2.imread(self.data_path/f"mask/{sample}.png", cv2.IMREAD_UNCHANGED))
        self.rgb = np.array(rgb)
        self.depth = np.array(depth)*self.splat_scale
        self.mask = np.array(mask)

  
    def get_detection(self):
        bb, uv = [], []
        for sample in self.sample_names:
            det = np.loadtxt(self.data_path/f"detection/{sample}.txt") 
            bb.append(det[:,:4].astype(np.int16))
            uv.append(det[:,4:6].astype(np.int16))
            
        self.bb = bb
        self.uv = uv


    def lift_to_3d(self):
        self.depth_val = []
        self.lifted_points = []
        for i in range(self.N):
            depth_val, _, _ = get_depth_value(
                self.bb[i], self.depth[i], self.mask[i], None, 0.1, 2.5, False
            )
            lifted_points_cam = get_points3d(self.uv[i], depth_val, self.K)
            cam_coord_dist = np.linalg.norm(lifted_points_cam, axis=1) 
            Pc = np.hstack((self.Rc[i], self.tc[i].reshape(-1,1)))
            lifted_points_cam_hom = np.hstack((lifted_points_cam, np.ones(lifted_points_cam.shape[0]).reshape(-1,1)))
            lifted_points = (Pc@lifted_points_cam_hom.T).T
            world_coord_dist = np.linalg.norm(lifted_points-self.tc[i], axis=1)

            self.lifted_points.append(lifted_points)
            self.depth_val.append(depth_val)


    def get_flower_model(self):
        #! Read splats
        points, colors = read_splats_ply(self.data_path/'splats.ply')

        #! Read transforms
        with open(self.splat_transforms_pth, 'r') as f:
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

        self.splat_scale = splat_scale
        self.splat_points = points
        self.splat_color = colors


    def get_splats_plot(self):
        point_cloud = go.Scatter3d(
            x=self.splat_points[:,0], 
            y=self.splat_points[:,1], 
            z=self.splat_points[:,2], 
            mode='markers',
            marker=dict(
                size=1,
                color=self.splat_color
            )
        )
        return [point_cloud]
    

    def get_lifted_points_plot(self):
        all_plots = []
        for i in range(self.N):
            lp = self.lifted_points[i]
            point_labels = np.arange(lp.shape[0])
            all_plots.append(go.Scatter3d(
                x=lp[:,0], 
                y=lp[:,1], 
                z=lp[:,2], 
                mode='markers',
                marker=dict(
                    size=3,      # Adjust size of points
                    color=self.plotly_color[i] # Assign RGB colors
                ),
                text=point_labels
            ))
        return all_plots

    
    def plot_3d(self, all_plots):
        fig = go.Figure(data=all_plots)
        fig.update_layout(
            title='3D Plot', 
            scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z')
        )
        fig.write_html("new_plot.html")
        print(f"3D plot written to: new_plot.html")

    
    def plot_rays(self, cam_center, points, color):
        """
        cam_center: (3,)
        points: (N,3)
        """
        all_rays = []
        for point in points:
            all_rays.append(go.Scatter3d(
                x=[cam_center[0], point[0]],
                y=[cam_center[1], point[1]],
                z=[cam_center[2], point[2]],
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
            ))
        return all_rays

 
    def plot_rays_all_cam(self, cam_tc, all_points, colors):
        all_rays_all_cam = []
        for i in range(cam_tc.shape[0]):
            all_rays_all_cam += self.plot_rays(cam_tc[i], all_points[i], colors[i])
        return all_rays_all_cam

 
    def project_back_points(self):
        self.proj_points = []
        for i in range(self.N):
            P = np.hstack((self.Rc[i], self.tc[i].reshape(-1,1)))
            P_4by4 = np.vstack((P, np.array([0,0,0,1])))
            ext = np.linalg.inv(P_4by4)[:3]
            points = np.hstack((self.splat_points, np.ones(self.splat_points.shape[0]).reshape(-1,1)))
            points_cam = (ext@points.T).T
            points_proj = (self.K@points_cam.T).T
            points_proj /= points_proj[:,-1].reshape(-1,1)
            points_proj = points_proj[:,:-1]
            self.proj_points.append(points_proj)


    def plot_projected_points(self, proj_points, canvas):
        for point,color in zip(proj_points, self.splat_color):
            x, y = int(point[0]), int(point[1])
            cv2.circle(canvas, (x, y), radius=3, color=color.tolist(), thickness=-1)


    def apply_depth_colormap(self, depth_clone):
        depth_clone = np.where(depth_clone>3000, 3000, depth_clone)
        depth_clone = (depth_clone-depth_clone.min())/(depth_clone.max()-depth_clone.min())
        depth_clone = depth_clone*255.0
        depth_clone = depth_clone.astype(np.uint8)
        depth_clone = cv2.applyColorMap(depth_clone, cv2.COLORMAP_JET)
        return depth_clone

    def plot_projected_points_all_cam(self):
        for i in range(self.N):
            rgb = self.rgb[i].copy()
            depth_clone = self.depth[i].copy()
            depth_clone = self.apply_depth_colormap(depth_clone)

            canvas = 0.5*depth_clone + 0.5*rgb

            self.plot_projected_points(self.proj_points[i], canvas)

            write_text_bottom(self.sample_names[i], canvas)
            plot_bounding_boxes(canvas, self.bb[i].astype(np.uint16))

            flower_labels = np.arange(self.bb[i].shape[0])

            for dv,bb,lb in zip(self.depth_val[i], self.bb[i], flower_labels):
                write_text(f"{lb}:{dv:.2f}", canvas, (int(bb[0]), int(bb[1]-20))   )


            cv2.imwrite(f"proj_{i}.png", canvas)
   
    
    def run(self):
        self.get_flower_model()
        self.get_sample_names()
        self.get_cam_poses()
        self.get_rgb_depth()
        self.get_detection()
        self.lift_to_3d()

        self.project_back_points()
        self.plot_projected_points_all_cam()

        splats_plot = self.get_splats_plot()
        axis_plot = plotly_axis()
        cam_plot = plotly_cam_poses(
            self.qctc, 
            scale=0.01, 
            plot_cam_count=False, 
            color=self.plotly_color,
            cam_names=self.sample_names
        )
        lifted_plot = self.get_lifted_points_plot()

        rays_plot = self.plot_rays_all_cam(self.tc, self.lifted_points, self.plotly_color)

        self.plot_3d(splats_plot+axis_plot+cam_plot+lifted_plot+rays_plot)
        
    
if __name__=='__main__':
    random.seed(88)
    np.random.seed(88)

    vis_data = VisDataset(
        path = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw',
        N = 15 # No. of vis
    )
    vis_data.run()