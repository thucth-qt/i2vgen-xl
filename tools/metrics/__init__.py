from .fvd_lib.calculate_fvd import calculate_fvd as calculate_fvd_lib
from .clip_multimodel import CLIPScore
from einops import rearrange
from torch.nn import functional as F
import torch
import gc

def get_key_frame(videos):
    # videos: BCTHW
    frames = videos[:,0,...] #BCHW
    return frames

def calculate_clipsim(videos, prompts, device):
    metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    metric.to(device)
    frames = get_key_frame(videos)
    score = metric(frames.to(device), prompts)
    score = score.detach().item()
    del metric 
    gc.collect()
    return score

def calculate_fvd(videos1, videos2, device):
    result = calculate_fvd_lib(videos1, videos2, device, method="videogpt")

    return result
