import numpy as np
import torch
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDINO():
    def __init__(self, device, text_prompt, box_th=0.2, text_th=0.3, obj_filter=None):
        self.device = device
        model_id = "IDEA-Research/grounding-dino-tiny"
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
        self.prompt = text_prompt
        self.box_th = box_th
        self.text_th = text_th
        self.obj_filter = obj_filter

 
    def detect(self, image):
        '''
        Args:
            image (np.ndarray): OpenCV image, BGR, (H,W,3)
        
        Returns:
            np.ndarary: (N,4) bounding boxes or #!(0,) if no bbox detected
        '''
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_th,
            text_threshold=self.text_th,
            target_sizes=[image.shape[:2]]
        )
        results = results[0]
       
        bounding_boxes = [] 
        for label,box in zip(results['labels'],results['boxes']):
            if self.obj_filter is not None and label != self.obj_filter:
                continue
            xmin,ymin,xmax,ymax = box.cpu().numpy().astype(np.int32)
            bounding_boxes.append([int(xmin), int(ymin), int(xmax), int(ymax)])
            
        return np.array(bounding_boxes)