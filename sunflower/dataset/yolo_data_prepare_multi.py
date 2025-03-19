import argparse
import random
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import shutil
from tqdm import tqdm

#! sunflower
from sunflower.utils.multi_aruco_pose_est import MultiArucoPoseEstimation
from sunflower.utils.mvg import project_3d_to_2d, get_bounding_box_from_reprojected_box
from sunflower.utils.plot import plot_projected_box_to_image
from sunflower.utils.image_manipulation import change_contrast
from sunflower.models.sam import SAM


if __name__=='__main__':
    """
    Prepare data to train YOLO model and also pose regression model. 
    It generates bounding boxes based on the pose estimated from aruco markers. 
    If no aruco markers are detected, then the image will be ignored.
    
    Two step process
    1. Generate Flower Pose Estimates and SAM mask for each image
        - Searches for all the *.jpg images inside input_dir/images
        - Generate flower pose estimate for all flowers in an image
        - Saves flower poses at input_dir/poses dir (with same dir struct as images)
        - Generates bounding boxes flowers as well using calculated GT poses
        - Generate mask using SAM and save it inside input_dir/masks
    2. Generate YOLO data point for each image, N times
    """
    #! Argument parser
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("-i", "--input", type=str, help="Path to input (raw) dir", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to partial YOLO Dataset", required=True)
    parser.add_argument("-r", "--random-seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()
    
    debug = True

    #! Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    #! Paths
    input_dir = Path(args.input) 
    if not input_dir.exists(): 
        print("Input dir doesn't exist")
        exit()
    output_dir = Path(args.output)
    
    images_path = input_dir/'images'
    masks_path = input_dir/'masks'
    generated_poses_path = input_dir/'poses'
    
    masks_path.mkdir(exist_ok=True, parents=True)
    generated_poses_path.mkdir(exist_ok=True, parents=True)
    
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
    
    
    #! Pre-defined flowers position in the aruco grid
    flower_positions = [
        (56.0, -56.0, 17.0),
        (120.0, -184.0, 17.0), 
        (-8.0, -184.0, 17.0), 
        (-8.0, 72.0, 17.0),
        (120.0, 72.0, 17.0)
    ]

    #! 3D bounding box for flower
    box3d = np.loadtxt(input_dir/'bounding_box_3d.txt')
   
    #! ----------------------- Part 1 -----------------------
    #! SAM
    sam = SAM('cuda')

    #! Images
    images_list = list(images_path.rglob("*.jpg"))
    images_list.sort()
    # images_list = images_list[7918:]
    images_list = images_list[7910:]
    nimages = len(images_list)
    print(f"{nimages} images available.")
    
    
    for img_path in tqdm(images_list):
        bounding_boxes = []
        poses = []
        projected_3d_boxes = []
        
        #! For each of 5 flowers
        for fpos in flower_positions:
            img_pil = Image.open(img_path)
            img_for_aruco = change_contrast(img_pil)
       
            #! Detect Aruco (redundant now)
            aruco = MultiArucoPoseEstimation(
                marker_size=48.0,
                marker_separation = 16.0,
                aruco_rows=6,
                aruco_columns=4,
                index_aruco=(2,1),
                aruco_to_origin=fpos,
                cam_intr=cam_intrinsics,
                aruco_dict=cv2.aruco.DICT_5X5_250,
                plot_marker=True,
                plot_pose=True
            ) 
            aruco_det = aruco.estimate_pose(img_for_aruco)
            if aruco_det is None:
                continue
            
            if debug:
                # img = cv2.imread(img_path)
                img = aruco_det['annotated_image']
                cv2.imwrite("frame.png", img)
            
            R,t = aruco_det['obj_R'], aruco_det['obj_tvec']
            q = aruco_det['obj_qvec']
            pose_data = f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f}\n"
            poses.append(pose_data)
            
            box2d = project_3d_to_2d(box3d, camera_matrix, R, t)
            projected_3d_boxes.append(box2d)
            
            crop_area = get_bounding_box_from_reprojected_box(box2d)
            bounding_boxes.append(crop_area)
            
    
        #! Filter out bounding boxes and poses
        bounding_boxes_new = []
        poses_new = []
        projected_3d_boxes_new = []
      
        margin = 5/100
        w,h = img_pil.size
        for bb,pp,p3db in zip(bounding_boxes, poses,projected_3d_boxes):
            xmin, ymin, xmax, ymax = bb
            
            if not (xmin > -w*margin and xmin < w*(1-margin)):
                continue
            if not (ymin > -h*margin and ymin < h*(1-margin)):
                continue
            if not (xmax > w*margin and xmax < w*(1+margin)):
                continue
            if not (ymax > h*margin and ymax < h*(1+margin)):
                continue
           
            bounding_boxes_new.append(bb)
            poses_new.append(pp)
            projected_3d_boxes_new.append(p3db)
           
        bounding_boxes = bounding_boxes_new 
        poses = poses_new
        projected_3d_boxes = projected_3d_boxes_new
        # exit()
        
         
        #! Get Mask    
        mask = sam.get_segmentation_mask(img_pil, bounding_boxes)
        
        #! Save Object Poses and Mask
        img_name_no_ext = str(img_path.relative_to(images_path))[:-3]
        poses_full_path = generated_poses_path/f"{img_name_no_ext}txt"
        Path(poses_full_path).parent.mkdir(exist_ok=True, parents=True)
        with open(poses_full_path, 'w') as fp:
            fp.writelines(poses)
        mask_full_path = masks_path/f"{img_name_no_ext}png" 
        Path(mask_full_path).parent.mkdir(exist_ok=True, parents=True)
        mask.save(mask_full_path) 
        
        if debug:
            for pbb in projected_3d_boxes:
                print('plotting projected box')
                plot_projected_box_to_image(img, pbb)
                
            # for bb in bounding_boxes:
            #     color = (255, 0, 0)
            #     thickness = 2
            #     xmin, ymin, xmax, ymax = bb
            #     cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, thickness)
                
            debug_full_path = masks_path/f"{img_name_no_ext}debug.png" 
            debug_full_path = "debug_nullify.png"
            cv2.imwrite(debug_full_path, img)
            
        exit()
            
                


    #! ----------------------- Part 2 -----------------------

    exit()
    # ntrain = int(len(images_list)*0.9)
    
    for i, img_name in enumerate(tqdm(images_list)):
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
        xmin, xmax, ymin, ymax = crop_area
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
