import torch
import numpy as np
from PIL import Image
from transformers import SamModel, SamProcessor
import cv2

class SAM():
    def __init__(self, device):
        self.device = device
        self.model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        
    def get_segmentation_mask(self, image, bounding_boxes):
        """
        Args:
            image: PIL.Image
            bounding_boxes: List[List[floats]]
                Each bounding in the from: [xmin, ymin, xmax, ymax]
        Returns:
            np.ndarray: Mask, (H, W), range (0, 255), np.uint8
        """
        # print(bounding_boxes)
        
        sam_inputs = self.processor(image, input_boxes=[bounding_boxes], return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**sam_inputs)
        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), 
            sam_inputs["original_sizes"].cpu(), 
            sam_inputs["reshaped_input_sizes"].cpu()
        )
        masks_np = masks[0].cpu().numpy() # (B,3,H,W) # boolean
        # it gives 3 types of mask: single instance, all instances, whole body
        # among that, we are interested in single instance only
        # which is the first masks out of 3 in the masks shape (B, >>3<< ,H,W)
        masks_np = masks_np[:,0,:,:]
        
        combined_mask = np.any(masks_np, axis=0)
        mask_uint8 = np.where(combined_mask==True, 255, 0).astype(np.uint8)
        
        # for bb in bounding_boxes:
        #     color = (255, 0, 0)  # Green color
        #     thickness = 2        # Thickness of the rectangle border
        #     xmin, ymin, xmax, ymax = bb
        #     cv2.rectangle(mask_uint8, (xmin, ymin), (xmax, ymax), color, thickness)
            
        mask_pil = Image.fromarray(mask_uint8)
        
        return np.array(mask_pil)

 
if __name__=='__main__':
    import os 
    from pathlib import Path
    from tqdm import tqdm

    path = Path('/home/rashik_shrestha/data/plantscan_pixel_1230')
    images = os.listdir(path/'rgb')
    images.sort()

    sam = SAM('cuda')
    print('sam loaded')
        
    for img in tqdm(images):
        fname = img[:-4]
        
        img_pil = Image.open(path/f"rgb/{fname}.jpg")
        detection = np.loadtxt(path/f"detection/{fname}.txt")
        
        bb = detection[:,:4]
        
        mask = sam.get_segmentation_mask(img_pil, bb.tolist())
        
        cv2.imwrite(path/f"mask/{fname}.png", mask)