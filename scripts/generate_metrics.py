import random
import os
from tqdm import tqdm
from pathlib import Path
import torch
from sunflower.dataset.posenet_flower_dataset import PoseNetFlowerDataset
from sunflower.models.posenet import PoseResNet
import roma
import cv2
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from ultralytics import YOLO
import pickle

from rich.console import Console
from rich.panel import Panel
console = Console()

#! Sunflower
from sunflower.utils.image_manipulation import change_contrast
from sunflower.utils.plot import plot_axis_and_translate, plot_bounding_boxes, plot_axis
from sunflower.utils.conversion import procrustes_to_rotmat
from sunflower.models.grounding_dino import GroundingDINO
from sunflower.models.sam import SAM
from sunflower.utils.data import get_pixel6a_cam_matrix, get_realsense_435_cam_matrix
from sunflower.utils.mvg import squarify_bb, bb_in_frame, filter_very_large_bb
from sunflower.utils.io import read_intrinsics_yaml_to_K_h_w, DatasetPath, pth
from sunflower.dataset.posenet_flower_dataset import PoseNetFlowerDataset
from sunflower.models.posenet import PoseResNet
from sunflower.utils.image_manipulation import get_depth_value
from generate_metrics_utils import *
from sunflower.utils.mvg import rot_average, get_points3d
from sunflower.dataset.flower_attn_dataset import FlowerAttnDataset

#! ----------------- config -----------------
max_count = 1000
seed = 0
device = 'cuda'
data_dir = '/home/rashik_shrestha/data/sunflower/plantscan_pixel_1230_raw'
posenet_weights = '/home/rashik_shrestha/ws/sunflower/scripts/weights/posenet_e183.pth'
yolo_weights = '/home/rashik_shrestha/ws/sunflower/output/yolo11nseg_1280.pt'
near_plane = 0.01
far_plane = 3.0
trans_th = 0.10
angle_th = 60
out_dir = Path('/home/rashik_shrestha/ws/sunflower/output/final_metrics_data')

#! Random seeds
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

#! Dataset
dataset = FlowerAttnDataset(data_dir)

#! Models
posenet_teacher = PoseResNet().to(device)
posenet_teacher.load_state_dict(torch.load(posenet_weights, weights_only=True))
print(f"✅ Posenet Loaded: {posenet_weights}")
gdino = GroundingDINO(device, 'white flower.', box_th=0.3, text_th=0.3, obj_filter='white flower')
print("✅ Grounding DINO Loaded")
sam = SAM(device)
print(f"✅ SAM loaded")
yolo = YOLO(yolo_weights).to(device)
print(f"✅ YOLO Loaded: {yolo_weights}\n")


#! Metrics Accumulator
total_gt_flowers = 0

gt_flowers_sucessfully_detected_by_teacher = []
gt_flowers_sucessfully_detected_by_student = []
teacher_det_errors = []
student_det_errors = []
teacher_to_student_det_errors = []

teacher_angle_errors = []
student_angle_errors = []
teacher_to_student_angle_errors = []

teacher_trans_errors = []
student_trans_errors = []
teacher_to_student_trans_errors = []

teacher_success_rates = []
student_success_rates = []

dice_all = []

count = -1
for data in tqdm(dataset):
    # Break Condition
    if count>max_count:
        print("Sequence Completed :)")
        break
    
    try:
        #! ---------- GT Data Visualization ----------
        img_tensor, mask_tensor, depth_tensor, poses, K, N = data
        if img_tensor is None:
            print("None data")
            continue
        # Count
        print('\n--------------------------------')
        count += 1 
        out_name = f"{count:05d}" 
        # Log
        log = open(out_dir/f"log/{out_name}.txt", "w")
        # Data Stats
        _,h,w = img_tensor.shape
        total_gt_flowers += N
        log.write(f"N: {N}\n")
        # print(img_tensor.shape, mask_tensor.shape, depth_tensor.shape, poses.shape, K.shape, N)
        # RGB
        img_pil: Image = to_pil_image(img_tensor)
        img_cv = np.array(img_pil) # OpenCV RGB
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR
        cv2.imwrite(out_dir/f"rgb/{out_name}.png", img_cv)
        # Depth
        depth = depth_tensor.numpy()
        depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        cv2.imwrite(out_dir/f"depth/{out_name}.png", depth_colormap)
        # Mask
        mask_gt = mask_tensor.numpy()
        mask_image = (mask_gt*255).astype(np.uint8)
        cv2.imwrite(out_dir/f"gt_mask/{out_name}.png", mask_image)
        # Intrin
        K = K.numpy()
        np.savetxt(out_dir/f"intrin/{out_name}.txt", K, fmt="%.3f")
        # Pose
        poses_gt = poses.numpy()[:N]
        trans_gt, rotmat_gt = poses_gt[:,:3,3], poses_gt[:,:3,:3]
        np.save(out_dir/f"gt_pose/{out_name}.npy", poses)
        pose_vis_img_gt = img_cv.copy()
        for R, t in zip(rotmat_gt, trans_gt):
            plot_axis(pose_vis_img_gt, R, t, K)
        cv2.imwrite(out_dir/f"gt_pose_vis/{out_name}.png", pose_vis_img_gt)
        
        #! ---------- Detection and Masking ----------
        # bounding box DINO
        bb_dino = gdino.detect(img_cv)
        log.write(f"bb dino: {bb_dino.shape}\n")
        if bb_dino.shape[0] == 0:
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
        bb_dino = filter_very_large_bb(bb_dino)
        log.write(f"bb dino after filter very large boxes: {bb_dino.shape}\n")
        if bb_dino.shape[0] == 0:
            log.write('bb_dino = 0 after filter very large bb')
            print('XXXXXXXXXXXXXXXXXXXXXXXXX')
        # uv DINO
        u_dino = (bb_dino[:,0]+bb_dino[:,2])/2
        v_dino = (bb_dino[:,1]+bb_dino[:,3])/2
        uv_dino = np.vstack((u_dino, v_dino)).T
        log.write(f"uv_dino: {uv_dino.shape}\n")
        # uv GT
        uv_gt = (K@trans_gt.T).T
        uv_gt /= uv_gt[:,-1].reshape(-1,1)
        uv_gt = uv_gt[:,:-1]
        log.write(f"uv_gt: {uv_gt.shape}\n")
        # bounding box YOLO
        bb_yolo, mask_yolo = get_yolo_bbox_mask(yolo, img_cv)
        log.write(f"bb_yolo: {bb_yolo.shape}\n")
        # uv YOLO
        u_yolo = (bb_yolo[:,0]+bb_yolo[:,2])/2
        v_yolo = (bb_yolo[:,1]+bb_yolo[:,3])/2
        uv_yolo = np.vstack((u_yolo, v_yolo)).T
        log.write(f"uv_yolo: {uv_yolo.shape}\n")
        # mask SAM
        mask_sam = sam.get_segmentation_mask(img_pil, bb_dino.tolist())
        # Plotting
        det_vis_img = img_cv.copy()
        plot_bounding_boxes(det_vis_img, bb_dino, (0,0,255))
        plot_2d_points(uv_dino, det_vis_img, (0,0,255))
        plot_bounding_boxes(det_vis_img, bb_yolo, (255, 0, 0))
        plot_2d_points(uv_yolo, det_vis_img, (255,0,0))
        plot_2d_points(uv_gt, det_vis_img, (0,255,0))
        cv2.imwrite(out_dir/f"vis_det_all/{out_name}.png", det_vis_img)
        cv2.imwrite(out_dir/f"sam_mask/{out_name}.png", mask_sam)
        cv2.imwrite(out_dir/f"yolo_mask/{out_name}.png", mask_yolo)
        
        #! ---------- Mask Evaluation ---------
        _, uv_dist = find_closest(uv_dino, uv_yolo)
        undetected_with_yolo = uv_dist > 10
        bb_dino_undet_with_yolo = bb_dino[undetected_with_yolo]
        mask_sam_from_yolos_pov = mask_sam.copy()
        for (xmin, ymin, xmax, ymax) in bb_dino_undet_with_yolo:
            mask_sam_from_yolos_pov[ymin:ymax, xmin:xmax] = np.zeros_like(mask_sam_from_yolos_pov[ymin:ymax, xmin:xmax])
        cv2.imwrite(out_dir/f"sam_mask_from_yolo_pov/{out_name}.png", mask_sam_from_yolos_pov)
        dice = dice_score(mask_yolo>128, mask_sam_from_yolos_pov>128)
        dice_all.append(dice)
        log.write(f"DICE: {dice:.3f}\n")
            

        #! ---------- Find which pred uv are closest to gt uv ----------    
        # Teacher
        uv_dino_idx, uv_gt_to_dino_dist = find_closest(uv_gt, uv_dino)
        uv_gt_to_dino_dist_filter = uv_gt_to_dino_dist<20
        log.write(f"No of bb_dino matched with gt: {uv_gt_to_dino_dist_filter.sum()}\n")
        gt_flowers_sucessfully_detected_by_teacher += [uv_gt_to_dino_dist_filter.sum()/uv_gt_to_dino_dist_filter.shape[0]]
        teacher_det_errors += uv_gt_to_dino_dist[uv_gt_to_dino_dist_filter].tolist()
        # Student 
        uv_yolo_idx, uv_gt_to_yolo_dist = find_closest(uv_gt, uv_yolo)
        uv_gt_to_yolo_dist_filter = uv_gt_to_yolo_dist<20
        log.write(f"No of bb_yolo matched with gt: {uv_gt_to_yolo_dist_filter.sum()}\n")
        gt_flowers_sucessfully_detected_by_student += [uv_gt_to_yolo_dist_filter.sum()/uv_gt_to_yolo_dist_filter.shape[0]]
        student_det_errors += uv_gt_to_yolo_dist[uv_gt_to_yolo_dist_filter].tolist()
        # Teacher -> Student
        uv_yolo_idx_for_dino, uv_dino_to_yolo_dist = find_closest(uv_dino, uv_yolo)
        uv_dino_to_yolo_dist_filter = uv_dino_to_yolo_dist<20
        log.write(f"No of bb_yolo matched with bb_dino: {uv_dino_to_yolo_dist_filter.sum()}\n")
        # gt_flowers_sucessfully_detected_by_teacher += uv_gt_to_dino_dist_filter.sum() # count the number of successful detection of GT flowers
        teacher_to_student_det_errors += uv_dino_to_yolo_dist[uv_dino_to_yolo_dist_filter].tolist()
        
        #! ---------- Pose Estimation ----------
        # Teacher
        uv_dino_matched_with_gt = uv_dino[uv_dino_idx][uv_gt_to_dino_dist_filter]
        bb_dino_matched_with_gt = bb_dino[uv_dino_idx][uv_gt_to_dino_dist_filter]
        log.write(f"bb dino to look for pose: {bb_dino_matched_with_gt.shape}\n")
        quat_teacher, rotmat_teacher, trans_teacher, orientaion_plot_teacher, patches_teacher, depth_reliable_teacher = pose_estimation(
            img_cv.copy(), depth.copy(), mask_sam, bb_dino_matched_with_gt, 
            posenet_teacher, uv_dino_matched_with_gt, K, device
        )
        # Student
        uv_yolo_matched_with_gt = uv_yolo[uv_yolo_idx][uv_gt_to_yolo_dist_filter]
        bb_yolo_matched_with_gt = bb_yolo[uv_yolo_idx][uv_gt_to_yolo_dist_filter]
        log.write(f"bb yolo to look for pose: {bb_yolo_matched_with_gt.shape}\n")
        quat_student, rotmat_student, trans_student, orientaion_plot_student, patches_student, depth_reliable_student = pose_estimation(
            img_cv.copy(), depth.copy(), mask_yolo, bb_yolo_matched_with_gt, 
            posenet_teacher, uv_yolo_matched_with_gt, K, device
        )
        # GT Orientation 
        rotmat_gt_matched_to_teacher = rotmat_gt[uv_gt_to_dino_dist_filter]
        quat_gt_matched_to_teacher = scipyR.from_matrix(rotmat_gt_matched_to_teacher).as_quat()
        rotmat_gt_matched_to_student = rotmat_gt[uv_gt_to_yolo_dist_filter]
        quat_gt_matched_to_student = scipyR.from_matrix(rotmat_gt_matched_to_student).as_quat()
        
        # Angle Error
        teacher_angle_error = quaternion_diff(quat_gt_matched_to_teacher, quat_teacher)
        student_angle_error = quaternion_diff(quat_gt_matched_to_student, quat_student)
        teacher_to_student_angle_error = quaternion_diff(quat_teacher, quat_student)
        log.write(f"teacher angle errors: {teacher_angle_error}\n")
        log.write(f"student angle errors: {student_angle_error}\n")
        log.write(f"teacher-student angle errors: {teacher_to_student_angle_error}\n")
        good_teacher_angle_error = teacher_angle_error<angle_th 
        good_student_angle_error = student_angle_error<angle_th 
        good_teacher_to_student_angle_error = teacher_to_student_angle_error<angle_th
        teacher_angle_errors += teacher_angle_error[good_teacher_angle_error].tolist()
        student_angle_errors += student_angle_error[good_student_angle_error].tolist()
        teacher_to_student_angle_errors += teacher_to_student_angle_error[good_teacher_to_student_angle_error].tolist()
        
        # Orientation Vis (GT matched with Student plot is omitted)
        orientaion_gt = plot_orientation_on_cropped_patches(patches_teacher, rotmat_gt_matched_to_teacher)
        orientaion_gt_plot = make_grid_cv(orientaion_gt) 
        orientaion_plot = np.vstack((orientaion_gt_plot, orientaion_plot_teacher, orientaion_plot_student))
        cv2.imwrite(out_dir/f"orient/{out_name}.png", orientaion_plot)
        
        # GT Trans
        #TODO: here if trans has unreliable depth, it is simple ignored and not reflected in metrics
        trans_gt_matched_to_teacher = trans_gt[uv_gt_to_dino_dist_filter]
        trans_gt_matched_to_student = trans_gt[uv_gt_to_yolo_dist_filter]
        
        teacher_trans_error =  np.linalg.norm(
            trans_gt_matched_to_teacher[depth_reliable_teacher]-trans_teacher[depth_reliable_teacher], 
            axis=1
        )
        student_trans_error =  np.linalg.norm(
            trans_gt_matched_to_student[depth_reliable_student]-trans_student[depth_reliable_student], 
            axis=1
        )
        depth_reliable_both = np.logical_and(depth_reliable_teacher, depth_reliable_student)
        teacher_to_student_trans_error =  np.linalg.norm(
            trans_teacher[depth_reliable_both]-trans_student[depth_reliable_both], 
            axis=1
        )
        log.write(f"teacher trans errors: {teacher_trans_error}\n")
        log.write(f"student trans errors: {student_trans_error}\n")
        log.write(f"teacher-student trans errors: {teacher_to_student_trans_error}\n") 
        good_teacher_trans_error = teacher_trans_error<trans_th
        good_student_trans_error = student_trans_error<trans_th
        good_teacher_to_student_trans_error = teacher_to_student_trans_error<trans_th
        teacher_trans_errors += teacher_trans_error[good_teacher_trans_error].tolist()
        student_trans_errors += student_trans_error[good_student_trans_error].tolist()
        teacher_to_student_trans_errors += teacher_to_student_trans_error[good_teacher_to_student_trans_error].tolist()
        
        # Pose success rate
        teacher_success = np.logical_and(good_teacher_angle_error, good_teacher_trans_error)
        teacher_success_rate = [np.sum(teacher_success)/teacher_success.shape[0]]
        teacher_success_rates += teacher_success_rate
        student_success = np.logical_and(good_student_angle_error, good_student_trans_error)
        student_success_rate = [np.sum(student_success)/student_success.shape[0]]
        student_success_rates += student_success_rate
        
        log.close()
    except:
        print("Exception Occoured")
        log.close()


#! Save Results
all_metrics = {
    'teacher_trans_error': np.array(teacher_trans_errors)*100,
    'student_trans_error': np.array(student_trans_errors)*100,
    'teacher_to_student_trans_error': np.array(teacher_to_student_trans_errors)*100,

    'teacher_angle_error': np.array(teacher_angle_errors),
    'student_angle_error': np.array(student_angle_errors),
    'teacher_to_student_angle_error': np.array(teacher_to_student_angle_errors),

    'teacher_success_rate': np.array(teacher_success_rates)*100,
    'student_success_rate': np.array(student_success_rates)*100,

    'teacher_det_error': np.array(teacher_det_errors),
    'student_det_error': np.array(student_det_errors),
    'teacher_to_student_det_error': np.array(teacher_to_student_det_errors),

    'teacher_det_success_rate': np.array(gt_flowers_sucessfully_detected_by_teacher)*100,
    'student_det_success_rate': np.array(gt_flowers_sucessfully_detected_by_student)*100,

    'dice': np.array(dice_all)
}

with open(out_dir/"metrics/all_metrics.pkl", "wb") as file:
    pickle.dump(all_metrics, file)
print(f'All metrics written to: all_metrics.pkl')


#! Output Results
msg = ""
# Trans Error
final_teacher_trans_error = np.mean(np.array(teacher_trans_errors))*100 #cm
final_student_trans_error = np.mean(np.array(student_trans_errors))*100 #cm
final_student_to_teacher_trans_error = np.mean(np.array(teacher_to_student_trans_errors))*100 #cm
msg += f"Trans Error: {final_teacher_trans_error:.2f} {final_student_trans_error:.2f} {final_student_to_teacher_trans_error:.2f}\n"
# Angle Error
final_teacher_angle_error = np.mean(np.array(teacher_angle_errors))
final_student_angle_error = np.mean(np.array(student_angle_errors))
final_student_to_teacher_angle_error = np.mean(np.array(teacher_to_student_angle_errors))
msg += f"Angle Error: {final_teacher_angle_error:.2f} {final_student_angle_error:.2f} {final_student_to_teacher_angle_error:.2f}\n"
# Pose Success Rate
final_teacher_success_rate = 100*np.mean(np.array(teacher_success_rates))
final_student_success_rate = 100*np.mean(np.array(student_success_rates))
msg += f"Success Rate: {final_teacher_success_rate:.2f} {final_student_success_rate:.2f}\n"
# Detection Error
final_teacher_detection_error = np.mean(np.array(teacher_det_errors))
final_student_detection_error = np.mean(np.array(student_det_errors))
final_teacher_to_student_detection_error = np.mean(np.array(teacher_to_student_det_errors))
msg += f"Det Error: {final_teacher_detection_error:.2f} {final_student_detection_error:.2f} {final_teacher_to_student_detection_error:.2f}\n"
# Detection Success Rate
final_gt_flowers_sucessfully_detected_by_teacher = 100*np.mean(np.array(gt_flowers_sucessfully_detected_by_teacher))
final_gt_flowers_sucessfully_detected_by_student = 100*np.mean(np.array(gt_flowers_sucessfully_detected_by_student))
msg += f"Det Success Rate: {final_gt_flowers_sucessfully_detected_by_teacher:.2f} {final_gt_flowers_sucessfully_detected_by_student:.2f}\n"
# DICE
dice_mean = np.mean(np.array(dice_all))
msg += f"Seg DICE: {dice_mean:.3f}"

console.print(Panel(msg, title="Results", expand=False, border_style="green"))