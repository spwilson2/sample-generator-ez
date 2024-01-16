import subprocess, os, sys, ipykernel
import os
from prefigure.prefigure import get_all_args
from copy import deepcopy
import math
from diffusion import sampling
import torch
from torch import optim, nn
from einops import rearrange
import torchaudio
from audio_diffusion.models import DiffusionAttnUnet1D

def createPath(filepath):
    os.makedirs(filepath, exist_ok=True)

class DiffusionUncond(nn.Module):
    def __init__(self, global_args):
        super().__init__()

        self.diffusion = DiffusionAttnUnet1D(global_args, n_attn_layers = 4)
        self.diffusion_ema = deepcopy(self.diffusion)
        self.rng = torch.quasirandom.SobolEngine(1, scramble=True)

def load_to_device(device, path, sr):
    audio, file_sr = torchaudio.load(path)
    if sr != file_sr:
        audio = torchaudio.transforms.Resample(file_sr, sr)(audio)
    audio = audio.to(device)
    return audio

def get_alphas_sigmas(t):
    """Returns the scaling factors for the clean image (alpha) and for the
    noise (sigma), given a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def get_crash_schedule(t):
    sigma = torch.sin(t * math.pi / 2) ** 2
    alpha = (1 - sigma ** 2) ** 0.5
    return alpha_sigma_to_t(alpha, sigma)

def t_to_alpha_sigma(t):
    """Returns the scaling factors for the clean image and for the noise, given
    a timestep."""
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)

def alpha_sigma_to_t(alpha, sigma):
    """Returns a timestep, given the scaling factors for the clean image and for
    the noise."""
    return torch.atan2(sigma, alpha) / math.pi * 2


from urllib.parse import urlparse
import hashlib
import k_diffusion as K


class Object(object):
    pass

args = get_all_args()
#args.ckpt_path = 'models/DanceDiffusion/bass-model-68/epoch=1404-step=440700.ckpt'#@param {type: 'string'}
#args.latent_dim = 0
#args.sample_rate = 48000 #@param {type: 'number'}
#args.sample_size = 65536 #@param {type: 'number'}
#args.batch_size = 2
# Number of times to create batch_size samples
#args.batch_count = 20
#args.demo_steps = 150
#args.sample_length_mult = 1
#args.sample_output_dir = os.curdir()
#args.sample_output_dir = 'generated'

try:
    createPath(args.sample_output_dir)
except:
    print(f'Unable to create sample_output_dir: {args.sample_output_dir}')
    args.sample_output_dir = os.curdir()
    print('Using as fallback f{args.sample_output_dir}')

print("Creating the model...")
model = DiffusionUncond(args)
model.load_state_dict(torch.load(args.ckpt_path, map_location=torch.device('mps'))["state_dict"])
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
model = model.requires_grad_(False).to(device)
print("Model created")

# # Remove non-EMA

# v-ddim | This is the sampler used in the training notebook. Needs more steps, but more reliable.
# v-iplms | Similar to above, but sometimes lower quality.
# k-heun | Needs fewer steps, but ideal sigma_min and sigma_max need to be found. Doesn't work with all models.
# k-dpmpp_2s_ancestral | Fastest sampler, but you may have to find new sigmas. Recommended min & max sigmas: 0.01, 80
# k-lms | "
# k-dpm-2 | "
# k-dpm-fast | "
# k-dpm-adaptive | Takes in extra parameters for quality, step count is non-deterministic
#@title Sampler options
sampler_type = "v-ddim" #@param ["v-ddim", "v-iplms", "k-heun", "k-dpmpp_2s_ancestral", "k-lms", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive"]
#sampler_type = "v-iplms" #@param ["v-ddim", "v-iplms", "k-heun", "k-dpmpp_2s_ancestral", "k-lms", "k-dpm-2", "k-dpm-fast", "k-dpm-adaptive"]

#@markdown **V-diffusion settings**
eta = 0 #@param {type: "number"}
#@markdown **K-diffusion settings**
sigma_min = 0.0001 #@param {type: "number"}
sigma_max = 1 #@param {type: "number"}
rho=7. #@param {type: "number"}
#@markdown k-dpm-adaptive settings
rtol = 0.01 #@param {type: "number"}
atol = 0.01 #@param {type: "number"}

def sample(model_fn, noise, steps=100, sampler_type="v-iplms", noise_level = 1.0):
    #Check for k-diffusion
    if sampler_type.startswith('k-'):
        denoiser = K.external.VDenoiser(model_fn)
        sigmas = K.sampling.get_sigmas_karras(steps, sigma_min, sigma_max, rho, device=device)

    elif sampler_type.startswith("v-"):
        t = torch.linspace(1, 0, steps + 1, device=device)[:-1]
        step_list = get_crash_schedule(t)

    if sampler_type == "v-ddim":
        return sampling.sample(model_fn, noise, step_list, eta, {})
    elif sampler_type == "v-iplms":
        return sampling.iplms_sample(model_fn, noise, step_list, {})

    elif sampler_type == "k-heun":
        return K.sampling.sample_heun(denoiser, noise, sigmas, disable=False)
    elif sampler_type == "k-lms":
        return K.sampling.sample_lms(denoiser, noise, sigmas, disable=False)
    elif sampler_type == "k-dpmpp_2s_ancestral":
        return K.sampling.sample_dpmpp_2s_ancestral(denoiser, noise, sigmas, disable=False)
    elif sampler_type == "k-dpm-2":
        return K.sampling.sample_dpm_2(denoiser, noise, sigmas, disable=False)
    elif sampler_type == "k-dpm-fast":
        return K.sampling.sample_dpm_fast(denoiser, noise, sigma_min, sigma_max, steps, disable=False)
    elif sampler_type == "k-dpm-adaptive":
        return K.sampling.sample_dpm_adaptive(denoiser, noise, sigma_min, sigma_max, rtol=rtol, atol=atol, disable=False)

def generate_from_noise(model_fn,
                        batch_size=4,
                        steps=100,
                        sample_length_mult=1):

    effective_length = sample_length_mult * args.sample_size

    # Generate random noise to sample from
    noise = torch.randn([batch_size, 2, effective_length]).to(device)

    generated = sample(model_fn, noise, steps, sampler_type)

    # Hard-clip the generated audio, fade between clips in the merged output
    fade_len = int(args.sample_rate*0.005)
    fade = torchaudio.transforms.Fade(
        fade_in_len=fade_len, 
        fade_out_len=fade_len, fade_shape="exponential")
    faded = fade(generated)
    generated_all = rearrange(faded, 'b d n -> d (b n)')
    generated = faded.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    return (generated, generated_all.clamp(-1, 1).mul(32767).to(torch.int16).cpu())

    # Put the demos together
    #generated_all = rearrange(generated, 'b d n -> d (b n)')
    #generated_all = generated_all.clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    #return generated_all

# Delete non-ema
del model.diffusion
def generate_once(idx):
    try:
        (generated, gen_all) = generate_from_noise(
            model_fn=model.diffusion_ema, 
            steps=args.demo_steps,
            sample_length_mult=args.sample_length_mult,
            batch_size=args.batch_size)
        
        filename = f'{args.name}-sum-{idx}.wav'
        filepath = os.path.join(args.sample_output_dir, filename)
        torchaudio.save(filepath, gen_all, args.sample_rate)
        for sidx, sample in enumerate(generated):
            filename = f'{args.name}-{idx}_{sidx}.wav'
            filepath = os.path.join(args.sample_output_dir, filename)
            torchaudio.save(filepath, sample, args.sample_rate)
        #torchaudio.save(filepath, generated_all, args.sample_rate)
    except KeyboardInterrupt:
        input('Press enter to goto next loop, or ctlr-c to quit.')

if args.batch_count < 0:
    i = 0
    while True:
        generate_once(i)
        i += 1
else:
    for i in range(args.batch_count):
        generate_once(i)