import io
import numpy as np
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

import comfy.utils
import comfy.samplers

def common_ksampler(model, seed, steps, scheduler, positive, latent, denoise=1.0, start_step=None, last_step=None):
    latent_image = latent["samples"]
    batch_inds = [0] * latent_image.shape[0]
    noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

    samples = comfy.sample.sample(
        model,
        noise,
        steps,
        cfg=1,
        sampler_name="euler",
        scheduler=scheduler,
        positive=positive,
        negative=positive,
        latent_image=latent_image,
        denoise=denoise,
        disable_noise=False,
        start_step=start_step,
        last_step=last_step,
        force_full_denoise=True,
        noise_mask=None,
        disable_pbar=True,
        seed=seed,
    )
    
    loss = F.mse_loss(samples, latent_image, reduction="none").mean(dim=(1,2,3))
    return loss

class MeasureLoss:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "The random seed used for creating the noise."}),
                "steps": ("INT", {"default": 30, "min": 3, "max": 1000, "tooltip": "The number of timesteps used to evaluate"}),
                "repeats": ("INT", {"default": 4, "min": 1, "max": 1000, "tooltip": "The number of times to repeat each step with new seeds (averaged)"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "The scheduler controls how noise is gradually removed to form the image."}),
                "conditioning": ("CONDITIONING", {"tooltip": "The conditioning describing the attributes you want to include in the image."}),
                "latent": ("LATENT", {"tooltip": "The latent image to test"}),
                "log_y_scale": ("BOOLEAN", {"default": False}),
                "limit_y_scale": ("FLOAT", {"default": 18, "min": 0.01, "step": 0.01}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("graph",)
    FUNCTION = "sample"
    CATEGORY = "LossTesting"
    DESCRIPTION = "Uses the provided model, conditioning, and encoded image to test loss across timesteps"
    
    def sample(self, model, seed, steps, repeats, scheduler, conditioning, latent, log_y_scale, limit_y_scale):
        losses = []
        timesteps = []
        pbar = comfy.utils.ProgressBar(steps) if comfy.utils.PROGRESS_BAR_ENABLED else None
        for i in tqdm(range(steps), desc="measuring loss"):
            loss = []
            for j in range(repeats):
                loss.append(common_ksampler(model, seed+j, steps, scheduler, conditioning, latent, denoise=1.0, start_step=i, last_step=i+1))
            loss = torch.stack(loss, dim=0).mean(dim=0)
            losses.append(loss)
            timesteps.append(i)
            if pbar is not None:
                pbar.update(1)
        
        plt.clf()
        for i in range(latent["samples"].shape[0]):
            img_loss = []
            for loss in losses:
                img_loss.append(loss[i].item())
            plt.plot(timesteps, img_loss, label=f"img {i}")
        
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        if log_y_scale:
            plt.yscale("log")
            lower_limit = 1e-2
        else:
            lower_limit = 0
        plt.ylim(lower_limit, limit_y_scale)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.clf()
        buf.seek(0)
        image = np.array(Image.open(buf))
        buf.close()
        
        image = torch.from_numpy(image[:, :, :3]).unsqueeze(0).float() / 255
        return (image,)
