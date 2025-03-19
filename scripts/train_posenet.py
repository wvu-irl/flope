import torch
import torch.nn as nn
from pathlib import Path
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import roma
import numpy as np
import cv2
from sunflower.utils.mvg import project_3d_to_2d

from sunflower.models.posenet import PoseResNet
from sunflower.dataset.posenet_flower_dataset import PoseNetFlowerDataset
from sunflower.utils.loss import diff_quats

#! Experimantal Settings
DEVICE = 'cuda'
dataset_path = "/home/rashik_shrestha/data/flower_posenet_data"
RANDOM_SEED = 0     # Helps to keep the experiments consistent
EPOCHS = 500          # Number of times to pass the entire dataset to the model
LR = 0.001            # learning rate
GAMMA = 0.1         # After each epoch, reduce the lerning rate to 70% of previous
BATCH_SIZE = 64
# WEIGHTS = '/home/rashik_shrestha/ws/sunflower/scripts/weights_r9t3/posenet_e212_r0.01302_a12.2_t0.00553.pth'
WEIGHTS = None
VIS_NUM = 24

#! Random
import random
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


#! Supporting Functions
def procrustes_to_rotmat(inp: torch.Tensor, **kwargs) -> torch.Tensor:
    return roma.special_procrustes(inp.reshape(-1, 3, 3))


def chordal_distance(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target).mean()

def get_annotated_image(image, R, t, K, gt=False):
    points = np.array([
        [0,0,0], [1,0,0], [0,1,0], [0,0,1]
    ])*0.05
    points2d = project_3d_to_2d(points, K, R, t)
    points2d = points2d.astype(np.int32)
   
    if gt: 
        cv2.line(image, points2d[0], points2d[1], color=(0,0,255), thickness=5)
        cv2.line(image, points2d[0], points2d[2], color=(0,255,0), thickness=5)
        cv2.line(image, points2d[0], points2d[3], color=(255,0,0), thickness=5)
    else:
        cv2.line(image, points2d[0], points2d[1], color=(0,0,255), thickness=2)
        cv2.line(image, points2d[0], points2d[2], color=(0,255,0), thickness=2)
        cv2.line(image, points2d[0], points2d[3], color=(255,0,0), thickness=2)
        
    
    # print(points2d)
    return image

def torch_to_numpy_image(img):
    """
    Args:
        img (torch.tensor): Torch Image, (C,H,W), float32, 0-1
    
    Returns:
        np.ndarray: Numpy Image, (H,W,C), unit8, 0-255
    """
    img = img.permute(1, 2, 0)
    img *= 255.0
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    return img


def plot_it(img, intrin, rot_gt, trans_gt, rot_pred, epoch):
    img, intrin, rot_gt, trans_gt, rot_pred = img[:VIS_NUM], intrin[:VIS_NUM], rot_gt[:VIS_NUM], trans_gt[:VIS_NUM], rot_pred[:VIS_NUM]
    img, intrin, rot_gt, trans_gt, rot_pred = img.detach().cpu(), intrin.detach().cpu().numpy(), rot_gt.detach().cpu().numpy(), trans_gt.detach().cpu().numpy(), rot_pred.detach().cpu().numpy()
   
    all_images = [] 
    for i in range(VIS_NUM):
        img_np = torch_to_numpy_image(img[i])
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        intrin_np = intrin[i]
        K = np.array([
            intrin_np[0], 0, intrin_np[2],
            0, intrin_np[1], intrin_np[3],
            0, 0, 1
        ])
        K = K.reshape(3,3)
        
        img_anno = get_annotated_image(img_np, rot_gt[i], trans_gt[i], K, gt=True)
        img_anno = get_annotated_image(img_np, rot_pred[i], trans_gt[i], K, gt=False)
        
        all_images.append(img_anno)
        # cv2.imwrite('prednew.png', img_anno)
       
    all_images = np.array(all_images)
    all_images = all_images.reshape(4,6,512,512,3)
    
    #! Make big image
    big_img = []
    for row_images in all_images:
        row = []
        for an_img in row_images:
            row.append(an_img)
        big_img.append(row)
   
    rows = [] 
    for row_images in big_img:
        rows.append(np.hstack(row_images))
    big_img = np.vstack(rows)
    
    # print(big_img.shape)
    cv2.imwrite(f"/home/rashik_shrestha/ws/sunflower/output/posenet/epoch{epoch:03d}.png", big_img)
    


def train_eval_single(data, model, optimizer, mode, vis=False, **kwargs):
    #! Data
    img, intrin, rot_gt, trans_gt = data
    img, intrin, rot_gt, trans_gt = img.to(DEVICE), intrin.to(DEVICE), rot_gt.to(DEVICE), trans_gt.to(DEVICE)
    
    #! Get Prediction
    if mode=='train': optimizer.zero_grad()
    r9_M_pred = model(img)
    rot_pred = procrustes_to_rotmat(r9_M_pred)
    
    #! Loss 
    loss = chordal_distance(rot_gt, rot_pred)
    
    quat_gt = roma.rotmat_to_unitquat(rot_gt)
    quat_pred = roma.rotmat_to_unitquat(rot_pred)
    dot_prod, angles = diff_quats(quat_gt, quat_pred)
    angle_error = torch.mean(angles)
    
    #! Backward
    if mode=='train':
        loss.backward()
        optimizer.step()
        
    if vis:
        plot_it(img, intrin, rot_gt, trans_gt, rot_pred, kwargs['epoch'])
  
    #! Return
    return loss.item(), angle_error.item()

def train_eval_epoch(epoch, dataloader, model, optimizer, mode):
    if mode=='train':
        model.train()
    else:
        model.eval()
        
    epoch_loss, epoch_angle_error = [], []
    for idx,data in enumerate(tqdm(dataloader)): 
        if idx==0 and mode=='eval': # Visualization Batch
            loss, angle_error = train_eval_single(data, model, optimizer, mode, vis=True, epoch=epoch)
        else:
            loss, angle_error = train_eval_single(data, model, optimizer, mode)
            
        epoch_loss.append(loss)
        epoch_angle_error.append(angle_error)

    epoch_loss = np.array(epoch_loss)
    epoch_angle_error = np.array(epoch_angle_error)
       
    return np.mean(epoch_loss), np.mean(epoch_angle_error)
    

def train_eval(train_loader, eval_loader, model, optimizer, scheduler):
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}:")
        #! Train
        train_epoch_loss, train_epoch_angle_error = train_eval_epoch(epoch, train_loader, model, optimizer, mode='train')
        
        #! Eval
        with torch.no_grad():
            eval_epoch_loss, eval_epoch_angle_error = train_eval_epoch(epoch, eval_loader, model, optimizer=None, mode='eval')
           
        log = f"{epoch} {train_epoch_loss:.5f} {train_epoch_angle_error:.2f} {eval_epoch_loss:.5f} {eval_epoch_angle_error:.2f}\n" 
        print(log)
        with open('posenet.log', 'a') as fp:
            fp.write(log)
        torch.save(model.state_dict(), f"weights/posenet_e{epoch}.pth")
      
 
if __name__=='__main__':
    #! Dataset
    train_dataset = PoseNetFlowerDataset(dataset_path, test=False)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataset = PoseNetFlowerDataset(dataset_path, test=True)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Train Loader Size: {len(train_dataloader)}")
    print(f"Test Loader Size: {len(eval_dataloader)}")

    #! Model
    model = PoseResNet().to(DEVICE)
    if WEIGHTS is not None:
        model.load_state_dict(torch.load(WEIGHTS, weights_only=True))

    #! Optimizer and Scheduler
    optimizer = optim.Adadelta(model.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=1, gamma=GAMMA) 
    
    #! Train, Eval
    print("Training Started...")
    train_eval(train_dataloader, eval_dataloader, model, optimizer, scheduler)