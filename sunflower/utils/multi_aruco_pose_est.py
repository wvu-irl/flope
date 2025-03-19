import numpy as np
import cv2

from sunflower.utils.conversion import rotmat2qvec, R2E, E2R

def get_aruco_offsets(nrows=5, ncolumns=4, offset_idx=(3,2)):
    """
    Get offsets index for a target aruco marker.
    nrows and ncolumns are the number of rows and columns in the AruCo matrix.
    offset_idx is the index of target AruCo marker, where (a,b) index represent
    ath row and bth column. counting starts from 0. 
    """
    rows = np.arange(nrows)
    columns = np.arange(ncolumns)

    x, y = np.meshgrid(columns, rows)
    x, y = x.flatten(), y.flatten()
    z = np.zeros_like(x)

    grid = np.vstack((x,y,z))
    grid  = grid.T

    grid[:,0] = offset_idx[1]-grid[:,0]
    grid[:,1] = -offset_idx[0]+grid[:,1]

    return grid


class MultiArucoPoseEstimation:
    def __init__(
        self, marker_size: float, marker_separation: float,
        aruco_rows, aruco_columns, index_aruco, aruco_to_origin,
        cam_intr: dict, aruco_dict, 
        plot_marker:bool = False, plot_pose:bool = False,
        aruco_max_id: int = 24
    ):
        """
        Pose estimation based on Matrix of AruCo markers.

        Parameters
        ----------
        marker_size: float
            Size (length/width) of Aruco marker in mm.
        marker_separation: float
            Separation between the markers in mm.
        index_aruco: Tuple(int)
            row and column of the index aruco. index starts from 0.
        aruco_to_origin: Tuple(float)
            (x,y,z) vector in milllimeters from index aruco top left to the object center.
        cam_intr:
            Dictionary consisting:
            {
                w: int
                h: int
                fx: float
                fy: float
                cx: float
                cy: float
                distortion: list[float]
            }
        aruco_dict:
            AruCo dictionary to use
        plot_marker: bool
            True if you want output plots of Markers
        plot_pose: bool
            True if you want output plot of Object Pose
        """
        #! Class Properties
        self.marker_size = marker_size/1000 # convert to meteres
        self.marker_separation = marker_separation/1000 # convert to meters
        self.plot_marker = plot_marker
        self.plot_pose = plot_pose
        self.marker_length = self.marker_size/2 # marker length is half the marker size
        self.aruco_max_id = aruco_max_id

        #! AruCo
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters()

        #! Calculate Target Object offsets w.r.t all the AruCo Markers
        aruco_sep = self.marker_size+self.marker_separation
        self.offsets = get_aruco_offsets(nrows=aruco_rows, ncolumns=aruco_columns, offset_idx=index_aruco)
        self.offsets = self.offsets*aruco_sep
        aruco_center = np.array([self.marker_size/2, -self.marker_size/2, 0])
        aruco_topleft_to_obj = np.array(aruco_to_origin)/1000     # For Flower
        self.offsets = self.offsets-aruco_center+aruco_topleft_to_obj

        #! Camera Intrinsics
        self.camera_matrix = np.array(
            [[cam_intr['fx'], 0, cam_intr['cx']],
             [0, cam_intr['fy'], cam_intr['cy']],
             [0, 0, 1]]
        )
        self.dist_coeffs = np.array(cam_intr['distortion'])


    def estimate_pose(self, image: np.array):
        """
        Detect Aruco Markers in given image.

        Parameters
        ----------
        image: np.array
            RGB Image of shape (H, W, 3)
            or Grayscale Image of shape (H, W)

        Returns
        -------
        Dictionary containing:
            ids: List[int] IDs of detected markers

        """
        # Convert to Grayscale if recieved RGB image
        if len(image.shape)==3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        # Detect AruCo Markers
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.parameters
        )

        # No markers available
        if ids is None:
            return None

        # Estimate the pose of each marker
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, self.marker_size, self.camera_matrix, self.dist_coeffs
        )

        # initial trans rot
        # init_trans = np.array([0.0,0.0,0.0]).reshape(-1,1)
        init_rot = np.array([0,0,0], dtype=np.float32)



        # tvecs[:,:,2] += 0.02
        # print(tvecs.shape)
        # exit()

        # Plot Markers
        if self.plot_marker:
            cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Plot Poses
        if self.plot_pose:
            rvecs_new = []
            tvecs_new = []
            for i in range(len(ids)):
                rvec, tvec = rvecs[i], tvecs[i]

                # Handle incorrect Aruco Detection
                if int(ids[i].item()) >= self.aruco_max_id:
                    continue

                init_trans = self.offsets[int(ids[i].item())].reshape(-1,1)

                Rvec, _ = cv2.Rodrigues(rvec)
                tvec_new = Rvec@init_trans + tvec.reshape(-1,1)
                tvec_new = tvec_new.reshape(1,1,3)

                init_R,_ = cv2.Rodrigues(init_rot)
                R_new = Rvec@init_R
                rvec_new,_ = cv2.Rodrigues(R_new)

                # print('rvec and tvec here:')
                # print(rvec_new.shape)
                # print(tvec_new.shape)

                # cv2.drawFrameAxes(
                #     image, self.camera_matrix, self.dist_coeffs, 
                #     rvec_new, tvec_new, length=self.marker_length
                # )

                rvecs_new.append(rvec_new)
                tvecs_new.append(tvec_new)
           
            if len(rvecs_new) == 0:
                return None

            rvecs_new = np.array(rvecs_new)
            tvecs_new = np.array(tvecs_new)

            rvecs_new_avg = np.median(rvecs_new, axis=0)
           
            
            tvecs_new_avg = np.median(tvecs_new, axis=0)

            Rvecs_new_avg, _ = cv2.Rodrigues(rvecs_new_avg)
            
            #! >> Nullify Yaw
            # euler_angles = R2E(Rvecs_new_avg)
            # euler_angles[0] = 0.0 # Nullify Yaw rotation
            # Rvecs_new_avg = E2R(euler_angles)
            # rvecs_new_avg,_ = cv2.Rodrigues(Rvecs_new_avg)
            #! << Nullify Yaw
           
            qvecs_new_avg = rotmat2qvec(Rvecs_new_avg.squeeze())

            cv2.drawFrameAxes(
                    image, self.camera_matrix, self.dist_coeffs, 
                    rvecs_new_avg, tvecs_new_avg, length=self.marker_length
                )
        
        # Prepare output data
        Rs = []
        for rvec in rvecs:
            R, _ = cv2.Rodrigues(rvec)
            Rs.append(R)
        Rs = np.array(Rs)

        out = {
            'corners': np.array(corners).squeeze(axis=1),
            'R': Rs,
            'rvec': rvecs.squeeze(axis=1),
            'tvec': tvecs.squeeze(axis=1),
            'annotated_image': image,
            'obj_R': Rvecs_new_avg.squeeze(),
            'obj_rvec': rvecs_new_avg.squeeze(),
            'obj_qvec': qvecs_new_avg,
            'obj_tvec': tvecs_new_avg.squeeze(),
        }
            
        return out


if __name__=='__main__':
    import os
    from pathlib import Path
    import torch
    import torchvision.transforms.functional as TF
    from PIL import Image
    from sunflower.utils.image_manipulation import change_contrast

    #! Camera Intrinsics (for pixel6a)
    cam_intrinsics = {
        'w': 1920,
        'h': 1080,
        'fx': 1751.276576,
        'fy': 1756.389162,
        'cx': 957.984186,
        'cy': 529.393387,
        'distortion': [0.0, 0.0, 0.0, 0.0, 0.0]
    }

    #! Input Image
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    # frame = cv2.imread(current_dir/'../../test_data/flower1.jpg')
    # frame = cv2.imread('/home/rashik_shrestha/ws/sunflower/output/frame_0000.png')
    frame = cv2.imread('/home/rashik_shrestha/data/flower_scan_undistort/flower1/frame_0000.jpg')

    # pilimage = Image.open('/home/rashik_shrestha/data/flower_scan_undistort/flower1/frame_0000.jpg')
    # frame = change_contrast(pilimage, 1.7)


    #! Multi Aruco Pose Estimator
    aruco = MultiArucoPoseEstimation(
        marker_size=48.0, # 54.5,
        marker_separation = 16.0, #5.45,
        aruco_rows=6, #5, 
        aruco_columns=4, #4, 
        index_aruco=(2,1), # (2,2), 
        # aruco_to_origin= (56.0, -56.0, 17.0), # (27.25, -27.25, 35), # (20.5, -47, 47) for box
        # aruco_to_origin= (120.0, -184.0, 17.0), # (27.25, -27.25, 35), # (20.5, -47, 47) for box
        # aruco_to_origin= (-8.0, -184.0, 17.0), # (27.25, -27.25, 35), # (20.5, -47, 47) for box
        # aruco_to_origin= (-8.0, 72.0, 17.0), # (27.25, -27.25, 35), # (20.5, -47, 47) for box
        aruco_to_origin= (120.0, 72.0, 17.0), # (27.25, -27.25, 35), # (20.5, -47, 47) for box
        cam_intr=cam_intrinsics,
        aruco_dict=cv2.aruco.DICT_5X5_250,
        plot_marker=True,
        plot_pose=True
    )

    #! Detect
    aruco_det = aruco.estimate_pose(frame)

    #! Print Outputs
    print("Output of ArucoMarker pose estimation:")
    if aruco_det is not None:
        for k,v in aruco_det.items():
            print(f"{k}: {v.shape}")
        cv2.imwrite(current_dir/f"../../output/obj_pose_using_aruco.png", aruco_det['annotated_image'])
        # cv2.imshow("Aruco Detection", aruco_det['annotated_image'])
        # cv2.waitKey(0)
    else:
        print("No Aruco Markers detected!")
