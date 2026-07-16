import os
import pickle
from typing import Union
import random

import numpy as np
import sys
import torch
from PIL import Image
from scipy import ndimage
import dlib



@torch.inference_mode()
def img2tensor(img: Union[np.ndarray, Image.Image]) -> torch.Tensor:
    """
    Converts a single PIL image to a tensor with mean [0.5, 0.5, 0.5] and std [0.5, 0.5, 0.5]
    """
    arr = np.asarray(img).astype(np.float32) / 127.5 - 1.0
    return torch.from_numpy(np.transpose(arr, (2, 0, 1))[np.newaxis, ...])


@torch.inference_mode()
def tensor2img(tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a single BCHW tensor to numpy images in BHWC format. Output images are cast to uint8 0-255
    """
    return (127.5 * (tensor + 1.0)).clamp(0, 255).permute(0, 2, 3, 1).cpu().to(torch.uint8).numpy()


def add_stylegan_repo(path: str = './stylegan2-ada-pytorch') -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f'{path} not found. Clone it first:\n'
            '  git clone https://github.com/NVlabs/stylegan2-ada-pytorch'
        )
    if path not in sys.path:
        sys.path.append(path)


def load_stylegan2_ffhq256(device, pkl='stylegan2-ffhq-256x256.pkl'):
    """Download (if missing) and load the FFHQ 256x256 StyleGAN2-ADA generator (EMA weights)."""
    if not os.path.exists(pkl):
        url = ("https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/stylegan2/1/"
               "files?redirect=true&path=stylegan2-ffhq-256x256.pkl")
        print('checkpoint not found, downloading...')
        os.system(f"wget --content-disposition '{url}' -O {pkl}")
        assert os.path.exists(pkl), 'download failed'
    with open(pkl, 'rb') as f:
        G = pickle.load(f)['G_ema'].to(device)
    return G.eval()
    

@torch.no_grad()
def sample_wplus(G, n, device, truncation_psi=None, seed=None, batch_size=8):
    """Sample n W+ latents, (n, G.num_ws, 512).

    truncation_psi=None draws a fresh random psi per batch, which spreads the samples across
    W+ instead of clustering them near w_avg — useful when collecting training data.

    NB: a *fresh* z is drawn every batch. (Reusing one z across iterations — easy to do by
    accident — makes every latent a copy of the same face and silently kills the diversity of
    whatever you train on top.)
    """
    gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
    if seed is not None:
        random.seed(seed)
    out = []
    for i in range(0, n, batch_size):
        k = min(batch_size, n - i)
        z = torch.randn([k, G.z_dim], generator=gen, device=device)
        psi = random.random() if truncation_psi is None else truncation_psi
        out.append(G.mapping(z, None, truncation_psi=psi, update_emas=False))
    return torch.cat(out, dim=0)[:n]


# FFHQ face alignment
def download_dlib_predictor(path='shape_predictor_68_face_landmarks.dat'):
    """Fetch dlib's 68-point facial landmark predictor."""
    if not os.path.exists(path):
        print('dlib landmark model not found, downloading...')
        os.system('wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        os.system('bzip2 -dk shape_predictor_68_face_landmarks.dat.bz2')
        assert os.path.exists(path), 'download failed'
    return dlib.shape_predictor(path)


def get_landmark(filepath, predictor):
    """68 facial landmarks, (68, 2)."""
    detector = dlib.get_frontal_face_detector()
    img = dlib.load_rgb_image(filepath)
    dets = detector(img, 1)
    if len(dets) == 0:
        raise ValueError(f'no face detected in {filepath}')
    shape = predictor(img, dets[0])
    return np.array([[p.x, p.y] for p in shape.parts()])


def align_face(filepath, predictor, output_size=256, transform_size=1024, enable_padding=True):
    """FFHQ-style align + crop, so a photo matches what StyleGAN2-FFHQ was trained on.
    Standard recipe (as in FFHQ / encoder4editing). Returns a PIL image."""

    lm = get_landmark(filepath, predictor)
    lm_eye_left, lm_eye_right = lm[36:42], lm[42:48]
    lm_mouth_outer = lm[48:60]

    # build an oriented crop rectangle from eye/mouth geometry
    eye_left, eye_right = np.mean(lm_eye_left, axis=0), np.mean(lm_eye_right, axis=0)
    eye_avg = (eye_left + eye_right) * 0.5
    eye_to_eye = eye_right - eye_left
    mouth_avg = (lm_mouth_outer[0] + lm_mouth_outer[6]) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    img = Image.open(filepath).convert('RGB')

    # shrink for speed
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(img.size[0] / shrink)), int(np.rint(img.size[1] / shrink)))
        img = img.resize(rsize, Image.LANCZOS)
        quad /= shrink
        qsize /= shrink

    # crop
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
            int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0),
            min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]

    # pad (with a blurred, mirrored border) if the crop runs off the image
    pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))),
           int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0),
           max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        yy, xx, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(
            1.0 - np.minimum(np.float32(xx) / pad[0], np.float32(w - 1 - xx) / pad[2]),
            1.0 - np.minimum(np.float32(yy) / pad[1], np.float32(h - 1 - yy) / pad[3]))
        blur = qsize * 0.02
        img += (ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0, 1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]

    img = img.transform((transform_size, transform_size), Image.QUAD,
                        (quad + 0.5).flatten(), Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), Image.LANCZOS)
    return img
