import argparse
import random
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import random
from torchvision.transforms import ColorJitter
from tqdm import tqdm

np.set_printoptions(precision=3, suppress=True)

from sunflower.utils.image_manipulation import change_contrast
import sunflower.utils.data as sdata
from sunflower.utils.multi_aruco_pose_est import MultiArucoPoseEstimation
from sunflower.models.grounding_dino import GroundingDINO
from sunflower.models.sam import SAM
from sunflower.utils.mvg import project_3d_to_2d, get_bounding_box_from_reprojected_box, nullify_yaw
from sunflower.utils.mvg import homography_transform
from sunflower. utils.conversion import rotmat2qvec

class PoseNetDataGenerator():
    def __init__(self, input, output, device='cuda'):
        self.input = Path(input)
        self.output = Path(output)
        self.device = device
      
        #! Data  
        self.cam_matrix = sdata.get_pixel6a_cam_matrix()
        self.obj_positions = sdata.get_obj_positions_on_aruco_grid()
        self.cam_intrinsics = sdata.get_pixel6a_intrinsincs()
        self.box3d = np.loadtxt(self.input/'bounding_box_3d.txt')
            
        #! Models 
        self.gdino = GroundingDINO(self.device, 'single white flower.', obj_filter='single white flower')
        self.sam = SAM(self.device)
       
        
    def get_object_poses(self, image):
        '''Given an image, get poses of all the objects'''
        Rs, ts = [], []
        for fpos in self.obj_positions:
            aruco = MultiArucoPoseEstimation(
                marker_size=48.0,
                marker_separation = 16.0,
                aruco_rows=6,
                aruco_columns=4,
                index_aruco=(2,1),
                aruco_to_origin=fpos,
                cam_intr=self.cam_intrinsics,
                aruco_dict=cv2.aruco.DICT_5X5_250,
                plot_marker=True,
                plot_pose=True
            ) 
            aruco_det = aruco.estimate_pose(np.copy(image))
            
            if aruco_det is None:
                continue
            
            Rs.append(aruco_det['obj_R'])
            ts.append(aruco_det['obj_tvec'])
            
        Rs, ts = np.array(Rs), np.array(ts)    
        return Rs, ts


    def get_projected_3d_boxes(self, Rs, ts):
        '''
        Args:
            Rs (np.ndarray): Rotations (Nx3x3)
            ts (np.ndarray): Translations (Nx3)
        '''
        projected_3d_boxes = []
        for R,t in zip(Rs, ts):
            box2d = project_3d_to_2d(self.box3d, self.cam_matrix, R, t)
            projected_3d_boxes.append(box2d)
        return np.array(projected_3d_boxes)
        

    def get_obj_bounding_boxes_using_aruco_poses(self, Rs, ts):
        '''
        Args:
            Rs (np.ndarray): Rotations (Nx3x3)
            ts (np.ndarray): Translations (Nx3)
        '''
        bounding_boxes = []
        projected_3d_boxes = self.get_projected_3d_boxes(Rs, ts)
        for p3b in projected_3d_boxes:
            bounding_boxes.append(get_bounding_box_from_reprojected_box(p3b))
        return np.array(bounding_boxes)
        

    def plot_bounding_boxes(self, image, bounding_boxes, color=(0,0,255)):
        for bb in bounding_boxes:
            xmin, ymin, xmax, ymax = bb
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color, 10)
        return image
    
   
    def filter_outside_bb(self, bounding_boxes, img_shape):
        '''
        Filter out bounding boxes that lies outside the image
        
        Args:
            bounding_boxes (np.ndarray): (N,4)
            img_shape (tuple(int)): (h,w)
        Returns:
            np.ndarray: (N,), boolean
        '''
        margin = 5/100
        h, w = img_shape
        
        # print('width and heights are')
        # print(h, w)
       
        bb_good = [] 
        bb_new = []
        for bb in bounding_boxes:
            xmin, ymin, xmax, ymax = bb
            
            if not (xmin > -w*margin and xmin < w*(1-margin)):
                bb_good.append(False)
                bb_new.append(None)
            elif not (ymin > -h*margin and ymin < h*(1-margin)):
                bb_good.append(False)
                bb_new.append(None)
            elif not (xmax > w*margin and xmax < w*(1+margin)):
                bb_good.append(False)
                bb_new.append(None)
            elif not (ymax > h*margin and ymax < h*(1+margin)):
                bb_good.append(False)
                bb_new.append(None)
            else:
                bb_good.append(True)
                bb_new.append(bb)
        
        return bb_new, np.array(bb_good)
       
       
    def get_bb_center(self, bb):
        x = (bb[0]+bb[2])/2
        y = (bb[1]+bb[3])/2
        return (x,y)
        
        
    def get_closest_bb(self, bb_ref, bb_options):
        # print(bb_ref)
        # print(bb_options)
        
        x_ref, y_ref = self.get_bb_center(bb_ref)
       
        best_dist = None 
        best_i = None
        for i,bbo in enumerate(bb_options):
            x, y = self.get_bb_center(bbo)
            
            dist = ((x_ref-x)**2 + (y_ref-y)**2)**(1/2)
            
            if best_dist == None or dist < best_dist:
                best_dist = dist
                best_i = i
        
        return bb_options[int(best_i)] 
        
       
    def map_aruco_to_dino_bb(self, aruco_bb, dino_bb):
        '''
        - dino_bb should lie inside aruco_bb (after added outside padding for aruco_bb)
        - if more than one dino_bb lies inside, select one with closest center
        - if one dino_bb lies inside aruco_bb, reject that data point
        '''
        new_dino_bb = []
        bb_good = []
       
        for bb in aruco_bb:
            if bb is None:
                new_dino_bb.append(None)
                bb_good.append(False)
                continue
            
            bb_close = []
            for bb2 in dino_bb:
                if bb2[0] > bb[0] and bb2[1] > bb[1] and bb2[2] < bb[2] and bb2[3] < bb[3]:
                    bb_close.append(bb2)
           
            if len(bb_close) == 0:
                new_dino_bb.append(None)
                bb_good.append(False)
            elif len(bb_close) == 1:
                new_dino_bb.append(bb_close[0])
                bb_good.append(True)
            else:  
                new_dino_bb.append(self.get_closest_bb(bb, bb_close))
                bb_good.append(True)
             
        return new_dino_bb, np.array(bb_good)
    
    
    def filter_out(self, bb_good, aruco_bb, dino_bb, Rs, ts):
        # print(bb_good.shape, Rs.shape, ts.shape)
        # print(aruco_bb, dino_bb)
      
        aruco_bb_new, dino_bb_new = [],[] 
        for i,bbg in enumerate(bb_good):
            if bbg:
                aruco_bb_new.append(aruco_bb[i])
                dino_bb_new.append(dino_bb[i])
        
        aruco_bb_new = np.array(aruco_bb_new) 
        dino_bb_new = np.array(dino_bb_new)
        Rs_new = Rs[bb_good]
        ts_new = ts[bb_good]
        
        return aruco_bb_new, dino_bb_new, Rs_new, ts_new
       

    def add_random_color_jitter(self, image, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        """
        Applies random color jitter to an OpenCV image using PyTorch's ColorJitter.
        
        Args:
            image (numpy.ndarray): Input image in OpenCV format (BGR).

        Returns:
            numpy.ndarray: Color-jittered image in OpenCV format (BGR).
        """
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        jitter = ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
        jittered_image = jitter(pil_image)
        jittered_image = cv2.cvtColor(np.array(jittered_image), cv2.COLOR_RGB2BGR)
        return jittered_image


    def add_homography_agu(self, image, mask, Rs, ts):
        # print(image.shape, mask.shape, Rs.shape, ts.shape)
        
        Rx = random.uniform(-10, 10)
        Ry = random.uniform(-10, 10)
        Rz = random.uniform(-180, 180)
        
        imageT, maskT, RsT, tsT, _ = homography_transform(
            image, (Rx,Ry, Rz), self.cam_matrix, mask, Rs, ts
        )
        
        return imageT, maskT, RsT, tsT
  
    
    def detect_obj_using_aruco_and_dino(self, img_cv, Rs, ts, save_filename=None):
        #! Detect dino and aruco bounding boxes
        bb_dino = self.gdino.detect(img_cv)
        bb_aruco = self.get_obj_bounding_boxes_using_aruco_poses(Rs, ts)
       
        #! Filter out Aruco boxes that are out of the frame 
        bb_aruco_inframe, bb_good_1 = self.filter_outside_bb(bb_aruco, img_cv.shape[:2])
        
        #! Map Aruco to Dino
        bb_dino_inside_bb_aruco, bb_good_2 = self.map_aruco_to_dino_bb(bb_aruco_inframe, bb_dino)
       
        #! Get filter that is good for both Aruco and Dino 
        bb_good = np.logical_or(bb_good_1, bb_good_2)
       
        #! Filter out everything 
        bb_aruco, bb_dino, Rs, ts = self.filter_out(
            bb_good,
            bb_aruco_inframe, bb_dino_inside_bb_aruco, 
            Rs, ts
        )
       
        #! Save both bounding boxes
        if save_filename is not None:
            img_cv_plot = self.plot_bounding_boxes(np.copy(img_cv), bb_dino, color=(0,0,255))
            self.plot_bounding_boxes(img_cv_plot, bb_aruco, color=(255,0,0))
            cv2.imwrite(save_filename, img_cv_plot)
        
        return bb_aruco, bb_dino, Rs, ts
    

    def squarify_bb(self, bb):
        xmin, ymin, xmax, ymax = bb
        xrange = xmax-xmin
        yrange = ymax-ymin
        diff = abs(xrange-yrange)
        if diff%2 == 0:
            decrease_min = diff/2
            increase_max = diff/2
        else:
            decrease_min = (diff+1)/2 # 
            increase_max = (diff-1)/2
        if xrange>yrange:
            ymin -= decrease_min
            ymax += increase_max
        elif xrange<yrange:
            xmin -= decrease_min
            xmax += increase_max
        
        return [int(xmin), int(ymin), int(xmax), int(ymax)]
            

    def bb_in_frame(self, bb, img_shape):
        h,w,_ = img_shape
        xmin, ymin, xmax, ymax = bb
        if xmin<0 or ymin<0 or xmax>w or ymax>h:
            return False
        else:
            return True

       
    def get_crop_intrinsics(self, offset, scale):
        '''
        Args:
            offset (tuple): (cropx, cropy)
        Returns:
            np.ndarray: Camera Intrinsic Matrix of cropped image, (3x3)
        '''
        crop_intrin = np.copy(self.cam_matrix)
        crop_intrin[0,2] -= offset[0]
        crop_intrin[1,2] -= offset[1]
        crop_intrin = scale*crop_intrin
        crop_intrin[2,2] = 1.0 #! Dont bloody hell forget this while scaling!!!!!!!!!!!
        return crop_intrin


    def get_annotated_image(self, image, R, t, K):
        points = np.array([
            [0,0,0], [1,0,0], [0,1,0], [0,0,1]
        ])*0.05
        points2d = project_3d_to_2d(points, K, R, t)
        points2d = points2d.astype(np.int32)
        
        cv2.line(image, points2d[0], points2d[1], color=(0,0,255), thickness=5)
        cv2.line(image, points2d[0], points2d[2], color=(0,255,0), thickness=5)
        cv2.line(image, points2d[0], points2d[3], color=(255,0,0), thickness=5)
        
        # print(points2d)
        return image

 
    def generate(self):
        print("Data Generation Started")
       
        #! Get Images list 
        images_list = list((self.input/'images').rglob("*.jpg"))
        images_list.sort()
        # images_list = images_list[7910:]
        nimages = len(images_list)
        print(f"{nimages} images available.")
        images_list += images_list
        
        count = 0
        errorcount = 0
        for img_path in tqdm(images_list):
            try:
                #! Get Image
                img_pil = Image.open(img_path) # PIL (RGB)
                img_high_contrast = change_contrast(img_pil) # OpenCV BGR
                img_cv = np.array(img_pil) # OpenCV RGB
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR) # OpenCV BGR
            
                #! Get Object poses using aruco 
                Rs, ts = self.get_object_poses(img_high_contrast)
            
                #! Detect Obj using Aruco and Dino 
                bb_aruco, bb_dino, Rs, ts = self.detect_obj_using_aruco_and_dino(
                    img_cv, Rs, ts, None #'aruco_detections.png'
                )
                
            
                #! Get Obj Mask 
                mask = self.sam.get_segmentation_mask(img_pil, bb_dino.tolist())
                # cv2.imwrite('mask.png', mask)
            
                #! Get Homography Transformed img, mask, Rs, ts 
                img_cvT, maskT, RsT, tsT = self.add_homography_agu(img_cv, mask, Rs, ts)
                # cv2.imwrite('maskT.png', maskT)
                
                #! Detect Obj using Aruco and Dino (for Homography Transformed)
                bb_arucoT, bb_dinoT, RsT, tsT = self.detect_obj_using_aruco_and_dino(
                    img_cvT, RsT, tsT, None #'aruco_detectionsT.png'
                )
                

                for bbt, Rt, tt in zip(bb_dinoT, RsT, tsT):
                    #! Squarify bb
                    bbt = self.squarify_bb(bbt)
                    xmin, ymin, xmax, ymax = bbt
                    
                    #! Ignore if bb out of frame
                    if not self.bb_in_frame(bbt, img_cvT.shape):
                        continue
                
                    #! Get square image patch
                    img_crop = img_cvT[ymin:ymax, xmin:xmax]
                    mask_crop = maskT[ymin:ymax, xmin:xmax]
                
                    img_crop_sized = cv2.resize(img_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
                    mask_crop_sized = cv2.resize(mask_crop, (512,512), interpolation=cv2.INTER_LANCZOS4)
                    
                
                    #! output name
                    out_name = f"{count:06d}"
                    
                    #! color jitter
                    img_crop_sized = self.add_random_color_jitter(img_crop_sized)
                    img_crop_sized_nobg = img_crop_sized * (mask_crop_sized.reshape(512,512,1)/255.0)
                    
                    cv2.imwrite(self.output/'images'/f"{out_name}.png", img_crop_sized_nobg) 
                    
                    
                    #! Get Crop Intrinsics
                    scale = 512/img_crop.shape[0]
                    crop_intrin = self.get_crop_intrinsics( (xmin, ymin), scale)
                    
                    with open(self.output/'intrinsics'/f"{out_name}.txt", 'w') as fp:
                        fp.write(f"{crop_intrin[0,0]:.4f}\t{crop_intrin[1,1]:.4f}\t{crop_intrin[0,2]:.4f}\t{crop_intrin[1,2]:.4f}")
                    
                
                    #! Get Extrinsics
                    Rt = nullify_yaw(Rt) # Rt is R transformed (dont get confused with rotation and translation)
                    qvec = rotmat2qvec(Rt)
                    with open(self.output/'poses'/f"{out_name}.txt", 'w') as fp:
                        fp.write(f"{qvec[0]:.6f}\t{qvec[1]:.6f}\t{qvec[2]:.6f}\t{qvec[3]:.6f}\t{tt[0]:.6f}\t{tt[1]:.6f}\t{tt[2]:.6f}")
                        
                    #! Get Annonated data
                    anno_img = self.get_annotated_image(img_crop_sized_nobg, Rt, tt, crop_intrin)
                    cv2.imwrite(self.output/'annotated_images'/f"{out_name}.png", anno_img) 
                    
                    #! Increase counter
                    count += 1
                    
                    # anno_img = self.get_annotated_image(np.copy(img_cvT), Rt, tt, self.cam_matrix)
                    # cv2.imwrite(self.output/'annotated_images'/f"{out_name}.full.png", anno_img) 
            except:
                print('Error count:', errorcount)
                errorcount += 1
                

if __name__=='__main__':
    """
    Generate Data to Train Flower PoseNet
    
    Output dir:
        images: (512,512) cropped and background removed images
        poses: (q,t) for each image
        intrinsics: (fx,fy,cx,cy) for each image
        annotated_images: uses calculated poses and intrinsics to annotate pose on image
    """
    #! Argument parser
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset")
    parser.add_argument("-i", "--input", type=str, help="Path to input (raw) dir", required=True)
    parser.add_argument("-o", "--output", type=str, help="Path to Output dir", required=True)
    parser.add_argument("-r", "--random-seed", type=int, help="Random seed", default=0)
    args = parser.parse_args()
    
    #! Random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    
    #! PoseNet Data generator
    data_generator = PoseNetDataGenerator(args.input, args.output)
    data_generator.generate()