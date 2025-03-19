import numpy as np

def torch_to_numpy_image(img):
    """
    Convert Torch (0-1) image to Numpy (0-255) image.
    Works for single Image.
    Works for both RGB and Grayscale image
    
    Args:
        img (torch.tensor): Torch Image, (C,H,W), float32, 0-1
    
    Returns:
        np.ndarray: Numpy Image, (H,W,C), unit8, 0-255
    """
    if len(img.shape)==3:
        img = img.permute(1, 2, 0)
    img *= 255.0
    img = img.detach().cpu().numpy()
    img = img.astype(np.uint8)
    return img


def numpy_to_torch_image(img):
    """
    Opposite of torch_to_numpy_image
    """
    return img