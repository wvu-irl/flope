import torch

def diff_quats(
    partices: torch.Tensor,  # (N,4)
    gt: torch.Tensor         # (N,4)
):
    """
    Args:
        particles (torch.Tensor): Particles quaternions
        gt (torch.Tensor): GT quaternions
    Returns:
        dot_pred (torch.Tensor): Dot prod of quats. Range of [-1,1]
        angle (torch.Tensor): Error angle in degrees. Range of [0,180]
    """
    dot_prod = torch.einsum("nd, nd -> n", partices, gt)
    dot_prod = torch.clamp(dot_prod, min=-1, max=1)
    angle = 2*torch.arccos(torch.abs(dot_prod)) * (180/torch.pi)
    return dot_prod, angle