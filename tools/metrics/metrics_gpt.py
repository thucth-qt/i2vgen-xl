import torch
import torchvision
import numpy as np
from torchvision.transforms import functional as F
from torchvision.models import inception_v3
from PIL import Image


def calculate_fid(video_path, num_samples=50000, batch_size=50):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained InceptionV3 model
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    # Load video frames
    video_frames = load_video_frames(video_path, num_samples)

    # Calculate FID
    real_features = calculate_inception_features(inception_model, video_frames, batch_size)
    fake_features = generate_fake_features(num_samples, batch_size, device)

    fid_score = calculate_fid_score(real_features, fake_features)
    return fid_score


def load_video_frames(video_path, num_samples):
    video = torchvision.io.read_video(video_path, num_frames=num_samples)[0]
    video_frames = video.permute(0, 3, 1, 2)  # [T, C, H, W]
    return video_frames


def calculate_inception_features(model, video_frames, batch_size):
    num_frames = video_frames.shape[0]
    features = []

    for i in range(0, num_frames, batch_size):
        batch_frames = video_frames[i:i + batch_size].to(device)
        batch_features = model(F.interpolate(batch_frames, size=(299, 299), mode='bilinear', align_corners=False))
        features.append(batch_features.view(batch_features.size(0), -1).detach().cpu().numpy())

    features = np.concatenate(features, axis=0)
    return features


def generate_fake_features(num_samples, batch_size, device):
    fake_features = torch.randn(num_samples, 2048).to(device)
    return fake_features


def calculate_fid_score(real_features, fake_features):
    # Calculate mean and covariance for real and fake features
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    cov_real = np.cov(real_features, rowvar=False)
    cov_fake = np.cov(fake_features, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_fake
    cov_mean, _ = scipy.linalg.sqrtm(cov_real.dot(cov_fake), disp=False)
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    fid_score = diff.dot(diff) + np.trace(cov_real) + np.trace(cov_fake) - 2 * np.trace(cov_mean)
    return fid_score


def calculate_clip_score(video_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pre-trained CLIP model
    clip_model, _ = clip.load("ViT-B/32", device=device)

    # Load video frames
    video_frames = load_video_frames(video_path, num_samples=10)

    # Calculate CLIP score
    clip_score = calculate_clip_average(clip_model, video_frames)
    return clip_score


def calculate_clip_average(clip_model, video_frames):
    video_frames = video_frames.permute(0, 2, 3, 1)  # [T, H, W, C]
    video_frames = (video_frames / 255.0).numpy()  # Normalize video frames to [0, 1]

    text_inputs = ["a photo of a cat", "a photo of a dog"]  # Example text inputs
    text_inputs = clip.tokenize(text_inputs).to(device)

    video_inputs = torch.from_numpy(video_frames).unsqueeze(0).to(device)
    video_inputs = video_inputs.permute(0, 4, 1, 2, 3).float()  # [B, T, C, H, W]

    with torch.no_grad():
        video_features = clip_model.encode_video(video_inputs)
        text_features = clip_model.encode_text(text_inputs)

    similarities = (100.0 * video_features @ text_features.T).softmax(dim=-1)
    average_similarity = similarities.mean().item()
    return average_similarity


# Example usage
video_path = "path/to/video.mp4"
fid_score = calculate_fid(video_path)
clip_score = calculate_clip_score(video_path)

print("FID Score:", fid_score)
print("CLIP Score:", clip_score)