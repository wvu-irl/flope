import argparse
import random
import numpy as np
from pathlib import Path
import cv2
import os
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm

#! sunflower
from sunflower.utils.multi_aruco_pose_est import MultiArucoPoseEstimation
from sunflower.utils.mvg import project_3d_to_2d, get_bounding_box_from_reprojected_box
from sunflower.utils.plot import plot_projected_box_to_image


if __name__=='__main__':
    """
    Prepare data to train YOLO model. It generated bounding boxes based on the
    pose estimated from aruco markers. If no aruco markers are detected, then
    the image will be ignored.
    """
    #! Argument parser
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("-i", "--input", type=str, help="Path to images dir", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to partial YOLO Dataset", required=True)
    parser.add_argument("-r", "--random-seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()

    #! Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    #! Paths
    input_dir = Path(args.input) 
    if not input_dir.exists(): print("Input dir doesn't exist")
    output_dir = Path(args.output)
    current_dir = Path(os.path.dirname(os.path.abspath(__file__)))

    train_img_path = output_dir/'images/train'
    val_img_path = output_dir/'images/val'
    train_label_path = output_dir/'labels/train'
    val_label_path = output_dir/'labels/val'
    train_pose_path = output_dir/'poses/train'
    val_pose_path = output_dir/'poses/val'

    train_img_path.mkdir(exist_ok=True, parents=True)
    val_img_path.mkdir(exist_ok=True, parents=True)
    train_label_path.mkdir(exist_ok=True, parents=True)
    val_label_path.mkdir(exist_ok=True, parents=True)
    train_pose_path.mkdir(exist_ok=True, parents=True)
    val_pose_path.mkdir(exist_ok=True, parents=True)

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

    camera_matrix = np.array([
        [cam_intrinsics['fx'], 0, cam_intrinsics['cx']],
        [0, cam_intrinsics['fy'], cam_intrinsics['cy']],
        [0, 0, 1]
    ])

    #! 3D bounding box for flower
    box3d = np.loadtxt(current_dir/'../../test_data/flower_3d_bounding_box.txt')

    #! Multi Aruco Pose Estimator
    aruco = MultiArucoPoseEstimation(
        marker_size=54.5,
        marker_separation = 5.45,
        aruco_rows=5, 
        aruco_columns=4, 
        index_aruco=(2,2), 
        aruco_to_origin=(27.25, -27.25, 35), # (20.5, -47, 47) for box
        cam_intr=cam_intrinsics,
        aruco_dict=cv2.aruco.DICT_5X5_250,
        plot_marker=True,
        plot_pose=True
    )


    images_list = os.listdir(input_dir)
    ntrain = int(len(images_list)*0.9)


    for i, img_name in enumerate(tqdm(images_list)):
        img = cv2.imread(input_dir/img_name)
        aruco_det = aruco.estimate_pose(img)
        if aruco_det is None:
            continue
        R,t = aruco_det['obj_R'], aruco_det['obj_tvec']
        q = aruco_det['obj_qvec']
        pose_data = f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f}"

        box2d = project_3d_to_2d(box3d, camera_matrix, R, t)
        # plot_projected_box_to_image(img, box2d)
        bbox, crop_area = get_bounding_box_from_reprojected_box(box2d)
        
        #! generate YOLO format data
        xmin, ymin, xmax, ymax = crop_area
        x_width = xmax-xmin
        y_width = ymax-ymin
        x_mid = xmin + x_width/2
        y_mid = ymin + y_width/2
        w, h = 1920, 1080
        x_width /= w
        y_width /= h
        x_mid /= w
        y_mid /= h
        data = f"0 {x_mid:.5f} {y_mid:.5f} {x_width:.5f} {y_width:.5f}\n"

        #! Move the img file
        if i<ntrain:
            shutil.move(input_dir/img_name, train_img_path)
            with open(train_label_path/f"{img_name[:-3]}txt", 'w') as fp:
                fp.write(data)
            with open(train_pose_path/f"{img_name[:-3]}txt", 'w') as fp:
                fp.write(pose_data)
        else:
            shutil.move(input_dir/img_name, val_img_path)
            with open(val_label_path/f"{img_name[:-3]}txt", 'w') as fp:
                fp.write(data)
            with open(val_pose_path/f"{img_name[:-3]}txt", 'w') as fp:
                fp.write(pose_data)

        # exit()
        # # fig, ax = plt.subplot()
        # plt.imshow(img)
        # plt.plot(bbox[:,0], bbox[:,1], color='blue')
        # plt.show()
        # exit()



    print("DONE")
