import torch
from fvd_lib.calculate_fvd import calculate_fvd
from clip_multimodel import CLIPScore
# ps: pixel value should be in [0, 1]!
if __name__ == "__main__":
    device = torch.device("cuda")
    
    NUMBER_OF_VIDEOS = 1
    VIDEO_LENGTH = 8
    CHANNEL = 3
    W = 448
    H = 256
    videos1 = torch.zeros(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H, W, requires_grad=False, device=device)
    videos2 = torch.ones(NUMBER_OF_VIDEOS, VIDEO_LENGTH, CHANNEL, H, W, requires_grad=False, device=device)

    import json
    result = {}
    result['fvd'] = calculate_fvd(videos1, videos2, device, method='styleganv')
    print(json.dumps(result, indent=4))


    # metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    # metric.to(device)
    # score = metric(videos1[0], ["a photo of a cat"]*videos1[0].shape[0])
    # score.detach()
    # print(score)
