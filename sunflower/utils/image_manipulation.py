import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import cv2
from numpy import ndarray
from jaxtyping import Float, jaxtyped, UInt8, Int, Bool
from beartype import beartype

from sunflower.utils.mvg import squarify_bb
from sunflower.utils.plot import apply_depth_colormap

def change_contrast(image, factor=1.5):
    tensor = TF.to_tensor(image)
    adjusted_tensor = TF.adjust_contrast(tensor, factor)
    adjusted_image = TF.to_pil_image(adjusted_tensor)
    frame = np.asarray(adjusted_image)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame


def shrink_mask(
    mask: Bool[ndarray, "H W"], 
    kernel_size: int = 3
) -> Bool[ndarray, "H W"]:
    """Shrinks the contour of a binary mask using erosion.

    Args:
        mask: Binary mask as a 2D NumPy array (True/False or 0/1).
        kernel_size: Size of the structuring element (integer).
    Returns:
        Eroded binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    eroded_mask = eroded_mask > 0 # Convert back to binary
    return eroded_mask


@jaxtyped(typechecker=beartype)
def get_depth_value(
    bbox: Int[ndarray, "N 4"], 
    depth: Float[ndarray, "H W"], 
    seg_mask: UInt8[ndarray, "H W"], 
    scale: float | None = None,
    near_plane: float = 0.1,
    far_plane: float = 3.0,
    vis: bool = False
) -> tuple[Float[ndarray, "N"], Bool[ndarray, "N"], Float[ndarray, "N 50 50"]|None]:
    """Get depth value
    
    Args: 
        bbox: Bounding box to extract depth data from depth image - xmin,ymin,xmax,ymax
        depth: Pixel value depth value in meters (normally loaded from .npy file)
        seg_mask: Segmenation mask Image 0=False, 255=True (normally read from uint8 grayscale image)
        scale: Depth scale factor
        near_plane: depth below this is bad
        far_plane: depth above this is bad
    Returns:
        Depth values in meters
        Depth Reliable Mask
        Depth Visualization or None
    """
    if scale: depth *= scale
    good_depth = np.logical_and(depth>near_plane,depth<far_plane) # near/far filter
    seg_mask = seg_mask>128 # Convert mask to boolean
    seg_mask = np.logical_and(seg_mask,good_depth)
    seg_mask = shrink_mask(seg_mask, 10)
    depth *= 1000 # depth to millimeters
    depth_values, depth_vis, depth_reliable = [], [], []
    for bb in bbox:
        wmin,hmin,wmax,hmax = bb
        depth_crop = depth[hmin:hmax, wmin:wmax]
        mask_crop = seg_mask[hmin:hmax, wmin:wmax]
        good_depths = depth_crop[mask_crop]
        # Less than 50 pixels of depth info is unreliable
        if good_depths.shape[0]<50:
            # print(good_depths.shape)
            depth_reliable.append(False)
        else:
            depth_reliable.append(True)
        # put depth=0 if no pixels found 
        if good_depths.shape[0]==0:
            depth_values.append(0)
        else:
            depth_values.append(np.mean(good_depths))
        if vis: 
            depth_vis_crop = np.where(mask_crop, depth_crop, 0)
            depth_vis_crop = cv2.resize(depth_vis_crop, (50,50))
            depth_vis.append(depth_vis_crop)
    depth_values =  np.array(depth_values)/1000 # Convert back to meters
    depth_reliable = np.array(depth_reliable)
    if vis: 
        depth_vis = np.array(depth_vis)
        return depth_values, depth_reliable, depth_vis
    else:
        return depth_values, depth_reliable, None
    
    
def detection_and_mask_to_contours(mask, bbox):
    """
    Args:
        mask: (H,W) uint8 Segmentation mask of 0s and 255s
        bbox: (N,4) np.ndarray of xmin,ymin,xmax,ymax
    Returns:
        List[List[float]] Contours
    """
    #! Find contours
    contours_raw, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #tuple of numpy array
    # Find center and area of each contours
    contours, contours_center, contours_area = [],[],[]
    for contour in contours_raw:
        # Calculate contour center
        M = cv2.moments(contour)
        if M["m00"] != 0:  # Avoid division by zero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            continue
        # Calculate contour area
        area = cv2.contourArea(contour)
        # Append to a list
        contours.append(contour)
        contours_center.append([cx,cy])
        contours_area.append(area)
    contours_center = np.array(contours_center) 

    #! For each bbox find the largest contour inside it
    #! might return less number of contours than number of bbox, if bbox overlapping
    bbox_contours = []
    contour_used = []
    for j, bb in enumerate(bbox):
        this_bb_contour, this_bb_contour_area = None, None
        xmin, ymin, xmax, ymax = bb
        # print(f"Finding contours for bb {j}: {xmin} {ymin} {xmax} {ymax}")
    
        # Check all the contours 
        for i, data in enumerate(zip(contours, contours_center, contours_area)):
            # Don't proceed if this contour is already assigned to a bbox
            if i in contour_used:
                continue
            
            con, conc, cona = data
            
            if conc[0]>xmin and conc[0]<xmax and conc[1]>ymin and conc[1]<ymax:
                # print(f"Countour center of {i} ({conc[0]}, {conc[1]}) lies inside bbox {j}")
                if this_bb_contour is None:
                    this_bb_contour = con
                    this_bb_contour_area = cona
                    contour_used.append(i)
                else:
                    if cona > this_bb_contour_area:
                        this_bb_contour = con
                        this_bb_contour_area = cona
                        contour_used.append(i)
                    
        if this_bb_contour is not None: 
            bbox_contours.append(this_bb_contour)
            
    return bbox_contours


def contours_to_polygons(contours, height, width):
    """Contours to polygons (normalized)
    """
    polygons = []
    for contour in contours:
        polygon = contour.flatten().tolist()
        normalized_polygon = [coord / width if i % 2 == 0 else coord / height for i, coord in enumerate(polygon)]
        polygons.append(normalized_polygon)
    return polygons