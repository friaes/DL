import random
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt

from einops import rearrange, repeat
from tqdm.notebook import tqdm
from matplotlib.animation import PillowWriter, FuncAnimation


def make_reproducible(seed=0):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_samples(ddpm, num_samples, image_shape):
    samples = []
    with torch.no_grad():
        x = torch.randn((num_samples,) + image_shape, device=ddpm.device)
        samples.append(x.cpu())
        
        for i in tqdm(range(ddpm.num_steps)):
            t = ddpm.num_steps - i - 1
            x = ddpm.p_sample(x, repeat(torch.tensor(t), '-> b', b=num_samples).to(dtype=torch.long, device=ddpm.device))
            samples.append(x.cpu())

    return samples


def generate_gif(samples_list, gif_path):
    figure, axes = plt.subplots()

    frames = get_frames(samples_list)
    animation = FuncAnimation(fig=figure, func=update, fargs=(axes, len(frames)), frames=frames, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=30))
    
    plt.close(figure)


def get_frames(samples_list):
    frames = []
    for i, samples in tqdm(enumerate(samples_list), total=len(samples_list)):
        samples_grid = torchvision.utils.make_grid(samples, nrow=int(np.sqrt(samples.shape[0])), normalize=True, pad_value=1)
        samples_grid = rearrange(samples_grid, 'c h w -> h w c')

        frames.append([samples_grid, i])

    return frames

def update(input_, axes, num_frames):
    frame, i = input_
    axes.clear()
    axes.axis('off')
    axes.set_title(f"t = {num_frames - i - 1}")
    axes.imshow(frame)
