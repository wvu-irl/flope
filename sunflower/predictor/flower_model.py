import numpy as np
from scipy.spatial.distance import cdist
from icecream import ic
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
# Need to change the matplotlib backend due to mismatched PyQt5 version with opencv
import matplotlib
matplotlib.use('tkagg')        

#! Sunflower imports
from sunflower.predictor.pose_predictor import PosePredictor
from sunflower.utils.mvg import pose_cam_to_world
from sunflower.utils.conversion import rotmat2qvec
from sunflower.utils.conversion import qvec2rotmat
from filterpy.kalman import KalmanFilter


def get_kalman_filter(initial_value):
    kf = KalmanFilter(dim_x=7, dim_z=7)
    kf.x = np.array(initial_value)  # Initial state estimate
    kf.F = np.eye(7)
    kf.H = np.eye(7)
    kf.P = np.eye(7)
    kf.Q = np.eye(7)*0.001
    kf.R = np.eye(7)*0.1 
    return kf

    
class FlowerModel():
    def __init__(self, 
        dist_th=50, 
        intrin_path='/home/rashik_shrestha/data/sunflower/flower_r405/intrinsics.yaml',
        get_plots=True
    ):
        self.get_plots = get_plots
        self.state = None
        self.scores = None
        self.kfs = []
        self.th = dist_th/1000

        self.pose_predictor = PosePredictor(
            device='cuda',
            posenet_path='/home/rashik_shrestha/ws/sunflower/scripts/weights/posenet_e183.pth',
            intrin_path=intrin_path
        )
       
        if self.get_plots: 
            # Initialize plot
            self.x = np.linspace(0, 10, 100)  # X-axis values
            self.y = np.sin(self.x)  # Initial Y-axis values
            
            F = 4
            self.F = F
            
            self.fig, self.axs = plt.subplots(F,7, figsize=(18,2*F))

            self.all_plots, self.all_x_data, self.all_y_data = [], [], []
            for f in range(F):
                plots, x_data, y_data = self.get_empty_plots(
                    self.axs[f], 
                    labels=['Mx', 'Sx', 'My', 'Sy', 'Mz', 'Sz', 'Mqx', 'Sqx', 'Mqy', 'Sqy', 'Mqz', 'Sqz', 'Mqw', 'Sqw'], 
                    linestyle=['dotted', 'solid']*7, 
                        color=2*['red']+2*['green']+2*['blue']+2*['orange']+2*['olive']+2*['purple']+2*['gray']
                )
                self.all_plots.append(plots)
                self.all_x_data.append(x_data)
                self.all_y_data.append(y_data)
            
            headings = ['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw']
            
            for f in range(F):
                for i in range(7):
                    self.axs[f, i].set_title(headings[i])
                    self.axs[f, i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                
            # self.ax.legend()
        
            plt.tight_layout() 
            plt.ion()  # Enable interactive mode
            plt.show()

 
    def get_empty_plots(self, axs, labels, linestyle, color):
        x = np.linspace(0, 10, 2)
      
        x_data = None 
        plots = [] 
        y_data = []
        
        for i, ax in enumerate(axs):
            # Measurement Plot
            lab,lstyle, col = labels[2*i], linestyle[2*i], color[2*i]
            plot, = ax.plot(x, x, label=lab, linestyle=lstyle, color=col)
            plots.append(plot)
            y_data.append(None)
            # State Plot
            lab,lstyle, col = labels[2*i+1], linestyle[2*i+1], color[2*i+1]
            plot, = ax.plot(x, x, label=lab, linestyle=lstyle, color=col)
            plots.append(plot)
            y_data.append(None)
            
            
        return plots, x_data, y_data
        
 
    def update_plot(self, new_data):
        """Simulate receiving new data and update the plot."""
        for data in new_data:
            state_idx, x_val, meas_value, updated_state_value = data

            # if state higher than 4, then so space to plot
            if state_idx >= self.F:
                continue
        
               
            if self.all_x_data[state_idx] is None:
                self.all_x_data[state_idx] = np.array([x_val])
                # Plot Measurements
                for i in range(7):
                    self.all_y_data[state_idx][2*i] = np.array([meas_value[i]])
                    self.all_y_data[state_idx][2*i+1] = np.array([updated_state_value[i]])
                
            else:
                self.all_x_data[state_idx] = np.concatenate((self.all_x_data[state_idx], np.array([x_val])), axis=0)
                for i in range(7):
                    self.all_y_data[state_idx][2*i] = np.concatenate((self.all_y_data[state_idx][2*i],np.array([meas_value[i]])), axis=0)
                    self.all_y_data[state_idx][2*i+1] = np.concatenate((self.all_y_data[state_idx][2*i+1],np.array([updated_state_value[i]])), axis=0)
                
            for i in range(14):
                self.all_plots[state_idx][i].set_xdata(self.all_x_data[state_idx])
                self.all_plots[state_idx][i].set_ydata(self.all_y_data[state_idx][i]) 
           
            for f in range(self.F):
                for i in range(7):
                    self.axs[f,i].relim()  # Recalculate limits based on new data
                    self.axs[f,i].autoscale_view()  # Rescale view to fit new data
                
       
        plt.draw()
        plt.pause(0.1)  # Pause to simulate real-time updates #TODO: yo navaye ni hunna ra?

        # plt.ioff()  # Disable interactive mode
        # plt.show()


    def assign_meas_to_state(self, meas):
        """
        The most important funtion.
        """
        ic('assign measurements to states')

        #! Plot data accumulator
        plot_data = []
               
        #! Initialize state and Filters (if the very first measurement) 
        if self.state is None:
            ic("Very first measurement")
            no_of_meas = meas.shape[0]
            self.state = meas
            self.scores = np.ones(no_of_meas)
            
            for idx in range(no_of_meas):
                each_meas = meas[idx]
                self.kfs.append(get_kalman_filter(each_meas))

                # Accumulate plot data
                if self.get_plots:
                    state_idx = idx
                    x_axis_timestamp = self.scores[idx]
                    meas_value = each_meas
                    updated_state_value = each_meas
                    plot_data.append([state_idx, x_axis_timestamp, meas_value, updated_state_value])
            
            # Update plot
            if self.get_plots:
                self.update_plot(plot_data)
                
        else:
            #! Calculate distances between current 
            state_trans = self.state[:,:3]
            meas_trans = meas[:,:3]
            distance_matrix = cdist(meas_trans, state_trans, metric='euclidean')
            min_idx = np.argmin(distance_matrix, axis=1)
            min_vals = np.min(distance_matrix, axis=1)
            good_matches = min_vals < self.th
            
            for i in range(meas.shape[0]):
                measurement = meas[i]
                # Update filter if it matches well
                if good_matches[i]:
                    ic(f"meas {i} matched to state {min_idx[i]}")
                    matched_state_idx = min_idx[i]
                    self.kfs[matched_state_idx].predict()
                    self.kfs[matched_state_idx].update(measurement)
                    self.kfs[matched_state_idx].x[3:] /= np.linalg.norm(self.kfs[matched_state_idx].x[3:])
                    self.scores[matched_state_idx] += 1
                   
                    # Accumulate plot data
                    if self.get_plots:
                        state_idx = matched_state_idx
                        x_axis_timestamp = self.scores[matched_state_idx]
                        meas_value = measurement
                        updated_state_value = self.kfs[matched_state_idx].x
                        plot_data.append([state_idx, x_axis_timestamp, meas_value, updated_state_value])
                
                # Else add it as a new state
                else:
                    ic(f"{i} didnt matched")
                    self.state = np.vstack((self.state, measurement.reshape(1,7)))
                    self.scores = np.hstack((  self.scores, np.array([1])  ))
                    self.kfs.append(get_kalman_filter(measurement))

            # Update Plot
            if self.get_plots:
                self.update_plot(plot_data)
                    
                
    def add_data(self, rgb, depth, cam_pose, ignore=False):
        """
        Args:
            rgb: (H,W,3)
            depth: (H,W)
            cam_pose: (7,) Poses in the form [3-translation,4-quaternion]
        Returns:
            flower_pose_cam: (N,4,4) Flower Poses in camera coordinate system
        """
        #! Cam Pose conversions
        cam_trans, cam_quat = cam_pose[:3], cam_pose[3:]
        cam_rotmat = qvec2rotmat(cam_quat)
        cam_posemat = np.hstack((cam_rotmat, cam_trans.reshape(3,1)))
        cam_posemat = np.vstack((cam_posemat, np.array([0,0,0,1])))
       
        #! Flower poses in camera coordinate system (YAW NULLIFIED)
        flower_pose_cam = self.pose_predictor.get_flower_poses(rgb, depth)
        
        if flower_pose_cam is None:
            return None, None
        
        # print(f"Flower[0] dist from cam (cam coordinate): {np.linalg.norm(flower_pose_cam[0,:3,3])}")
        
        #! Flower poses in world coordinate system
        flower_pose = pose_cam_to_world(flower_pose_cam, cam_posemat)
       
        # Extract flower trans and quaternions 
        flower_trans = flower_pose[:,:3,3]
        flower_rotmat = flower_pose[:,:3,:3]
        flower_quat = rotmat2qvec(flower_rotmat)
        meas = np.hstack((flower_trans, flower_quat))

        # print(f"Flower[0] dist from cam (world coordinate): {np.linalg.norm(flower_pose[0,:3,3]-cam_trans)}")

        if ignore: 
            self.assign_meas_to_state(meas)

        return flower_pose_cam, flower_pose.astype(np.float32)
        
        
    def get_state(self):
        return self.state

 
if __name__=='__main__':
    from sunflower.utils.io import DatasetPath, pth, read_intrinsics_yaml_to_K_h_w
    from tqdm import tqdm
    import cv2
    
    #! Configs
    data = DatasetPath('/home/rashik_shrestha/data/sunflower/flowerur')
    distance_th = 50
    score_th = 100
   
    model = FlowerModel(dist_th=distance_th) 
   
    # files = data.files[::10] 
    files = data.files
    
    for file in tqdm(files):
        ic(file)
        #! Read data
        image = cv2.imread(pth(data.rgb,file,'png'))
        depth = cv2.imread(pth(data.depth,file,'png'), cv2.IMREAD_UNCHANGED) #! Deoth values in mm
        cam_pose = np.loadtxt(pth(data.pose,file,'txt'))
        
        # ic(image.shape, image.dtype, image.min(), image.max())
        # ic(depth.shape, depth.dtype, depth.min(), depth.max())
        # ic(cam_pose.shape, cam_pose.dtype)
        
        #! Update model
        model.add_data(image, depth, cam_pose)
        ic(model.state.shape)
        print('done') 
        # input()