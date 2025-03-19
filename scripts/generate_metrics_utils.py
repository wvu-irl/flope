import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as scipyR
from scipy.spatial import cKDTree
from torchvision.utils import make_grid, save_image


#! sunflower
from sunflower.utils.conversion import procrustes_to_rotmat
from sunflower.utils.mvg import squarify_bb, bb_in_frame, filter_very_large_bb
from sunflower.utils.plot import plot_flower_poses_on_image
from sunflower.utils.image_manipulation import get_depth_value
from sunflower.utils.mvg import rot_average, get_points3d, nullify_yaw_batch


def prepare_data_for_posenet(image, mask, square_bbox, device='cuda'):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
    image_batch_np = [] 
    for bb in square_bbox: 
        # Get Image Patch Crops
        xmin, ymin, xmax, ymax = bb
        img_crop = image[ymin:ymax, xmin:xmax]
        mask_crop = mask[ymin:ymax, xmin:xmax]
        # Resize Crops to (512,512) 
        img_crop_sized = cv2.resize(img_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
        mask_crop_sized = cv2.resize(mask_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
        # Remove crop bg using the mask 
        img_crop_sized_nobg = img_crop_sized * (mask_crop_sized.reshape(512,512,1)/255.0)
        image_batch_np.append(img_crop_sized_nobg)
    # Accumulate batch 
    image_batch_np = np.array(image_batch_np)/255.0
    image_batch = torch.as_tensor(image_batch_np, dtype=torch.float32)
    image_batch = torch.permute(image_batch, (0,3,1,2)).to(device)
    return image_batch


def make_grid_cv(images, nrow=8, padding=2, pad_value=255):
    """
    Create a grid of images similar to torchvision.utils.make_grid but for OpenCV (NumPy) images.
    
    Args:
        images (list of np.ndarray): List of OpenCV images (H, W, C) in BGR format.
        nrow (int): Number of images per row.
        padding (int): Padding between images.
        pad_value (int): Padding color value (default: white).
    
    Returns:
        np.ndarray: Image grid in OpenCV BGR format.
    """
    # Ensure images have the same shape
    h, w, c = images[0].shape
    assert all(img.shape == (h, w, c) for img in images), "All images must have the same shape"

    num_images = len(images)
    ncol = (num_images + nrow - 1) // nrow  # Compute number of columns

    # Create a blank grid with padding
    grid_h = ncol * h + (ncol - 1) * padding
    grid_w = nrow * w + (nrow - 1) * padding
    grid = np.full((grid_h, grid_w, c), pad_value, dtype=np.uint8)

    # Place images in the grid
    for idx, img in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        y_start = row * (h + padding)
        x_start = col * (w + padding)
        grid[y_start:y_start + h, x_start:x_start + w, :] = img

    return grid


def get_quaternions_using_posenet(posenet, image_patches):
    r9_M_pred = posenet(image_patches) 
    rot_pred = procrustes_to_rotmat(r9_M_pred)
    rot_pred_np = rot_pred.detach().cpu().numpy() 
    rot_pred_np = nullify_yaw_batch(rot_pred_np)
    rot_pred_quat = scipyR.from_matrix(rot_pred_np).as_quat()
    return rot_pred_quat, rot_pred_np

def find_closest(gt_det, pred_det):
    tree = cKDTree(pred_det)
    distances, indices = tree.query(gt_det)
    return indices, distances


def quaternion_diff(q1, q2):
    """
    Compute the angle between two sets of quaternions (N,4).
    
    Parameters:
    q1: np.ndarray of shape (N,4) - First set of quaternions
    q2: np.ndarray of shape (N,4) - Second set of quaternions
    
    Returns:
    angles: np.ndarray of shape (N,) - Angles in degrees between corresponding quaternions
    """
    # Normalize quaternions to ensure unit norm
    q1 = q1 / np.linalg.norm(q1, axis=1, keepdims=True)
    q2 = q2 / np.linalg.norm(q2, axis=1, keepdims=True)

    # Compute dot product between corresponding quaternions
    dot_product = np.sum(q1 * q2, axis=1)

    # Ensure dot product is within valid range for acos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute the angle in radians and convert to degrees
    angles = 2 * np.arccos(dot_product) * (180.0 / np.pi)
    
    return angles

def get_yolo_bbox_mask(yolo, image):
    """
    Returns flower detection bounding boxes and segmentation mask
    """
    H,W,_ = image.shape
    results = yolo(image)
    masks = results[0].masks.data
    mask = torch.sum(masks, axis=0)
    mask = torch.clip(mask, 0, 1)*255
    mask = mask.cpu().numpy().astype(np.uint8)
    mask = cv2.resize(mask, (W,H))
    bbox = results[0].boxes.xyxy  # Bounding boxes in [x1, y1, x2, y2] format
    bbox = bbox.cpu().numpy().astype(np.int16)
    return bbox, mask


def plot_2d_points(points, image, color, radius=5):
    for (x, y) in points.astype(np.int32):
        cv2.circle(image, (x, y), radius=radius, color=color, thickness=-1)
        

def dice_score(mask1, mask2):
    """
    Compute the DICE score between two binary masks.
    
    Parameters:
        mask1 (numpy.ndarray): Binary mask of shape (H, W)
        mask2 (numpy.ndarray): Binary mask of shape (H, W)

    Returns:
        float: DICE score between the two masks.
    """
    assert mask1.shape == mask2.shape, "Masks must have the same shape"
    
    intersection = np.sum(mask1 * mask2)  # Logical AND
    total_pixels = np.sum(mask1) + np.sum(mask2)
    
    if total_pixels == 0:  # If both masks are empty, define DICE as 1.0
        return 1.0
    
    return 2 * intersection / total_pixels

def get_square_bb(bbox, image_shape):
    sq_bb = []
    for bb in bbox:
        sbb = squarify_bb(bb)
        if not bb_in_frame(sbb, image_shape):
            #TODO: Shifting the bbox to a side here, this might reduce the posenet accuracy!
            img_h, img_w, _ = image_shape
            if sbb[0] < 0:
                sbb[2] -= sbb[0]
                sbb[0] = 0
            if sbb[1] < 0:
                sbb[3] -= sbb[1]
                sbb[1] = 0
            if sbb[2] > img_w:
                sbb[0] -= sbb[2]-img_w
                sbb[2] = img_w
            if sbb[3] > img_h:
                sbb[1] -= sbb[3]-img_h
                sbb[3] = img_h
        sq_bb.append(sbb)
    return np.array(sq_bb)

def tensor_to_opencv(tensor_img):
    """
    Convert a PyTorch tensor image (C, H, W) to an OpenCV BGR image.
    
    Args:
        tensor_img (torch.Tensor): PyTorch tensor image with shape (C, H, W).
    
    Returns:
        np.ndarray: OpenCV image in BGR format.
    """
    # Ensure the tensor is on CPU and detach if it's from a computation graph
    tensor_img = tensor_img.cpu().detach()
    
    # Convert (C, H, W) → (H, W, C)
    np_img = tensor_img.permute(1, 2, 0).numpy()
    
    # Scale the image from [0,1] to [0,255] if necessary
    if np_img.max() <= 1:
        np_img = (np_img * 255).astype(np.uint8)
    else:
        np_img = np_img.astype(np.uint8)
    
    # Convert from RGB to BGR for OpenCV
    opencv_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
    
    return opencv_img


def plot_orientation_on_cropped_patch(patch, rotmat):
    tt = np.array([0,0,0.4])
    crop_intrin = np.array([
        [2433,0,256],
        [0,2433,256],
        [0,0,1]
    ])
    
    Rt = np.vstack((np.hstack((rotmat, tt.reshape(-1,1))), np.array([0,0,0,1])))
    plot_flower_poses_on_image(patch, Rt[None], crop_intrin, plot_count=False, plot_distance=False)
    
   
def plot_orientation_on_cropped_patches(patches, rotmats):
    """
    patches: (BCHW) tensor RGB format
    rotmats: (B33) tensor
    """
    plotted_patches = []
    for patch, R in zip(patches, rotmats):
        patch_cv = tensor_to_opencv(patch)
        plot_orientation_on_cropped_patch(patch_cv, R)
        
        plotted_patches.append(patch_cv)
        
    return np.array(plotted_patches)
   
 
def pose_estimation(img, depth, mask, bbox, posenet, uv, K, device):
    sq_bb = get_square_bb(bbox, img.shape)
    image_batch = prepare_data_for_posenet(img, mask, sq_bb, device) 
    image_batch_plot = make_grid(image_batch)
    save_image(image_batch_plot, 'patches.png')
    
    quat, rotmat = get_quaternions_using_posenet(posenet, image_batch)
    
    plotted_image_batch = plot_orientation_on_cropped_patches(image_batch, rotmat)
    plotted_image_batch_plot = make_grid_cv(plotted_image_batch)
    
    #! Get depth values
    depth_val, depth_reliable, _ = get_depth_value(
        bbox, depth, mask,
            near_plane = 0.1, far_plane = 3.0
    )
    
    trans = get_points3d(uv, depth_val, K)
    
    return quat, rotmat, trans, plotted_image_batch_plot, image_batch, depth_reliable