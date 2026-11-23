"""Reconstruction, Fourier, windowed spectrum and local morphology terms."""

import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models


def gradient_loss_spatial(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[:, :, :, 1:] - x[:, :, :, :-1]
        gy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return gx, gy

    gx1, gy1 = grad(pred)
    gx2, gy2 = grad(target)
    return F.l1_loss(gx1, gx2) + F.l1_loss(gy1, gy2)


def fft_magnitude_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pf = torch.fft.fft2(pred, dim=(-2, -1))
    tf = torch.fft.fft2(target, dim=(-2, -1))
    return F.l1_loss(torch.abs(pf), torch.abs(tf))


class VGG19PerceptualLoss(nn.Module):
    """
    ImageNet VGG19 五段特征（浅→深）加权 L1；权重 [1/32, 1/16, 1/8, 1/4, 1]。
    若环境变量 FM_VGG19_WEIGHTS 为本地 .pth 路径（与官方 vgg19-dcbb9e9d.pth 一致），则不从网络下载。
    """

    def __init__(self) -> None:
        super().__init__()
        local = os.environ.get("FM_VGG19_WEIGHTS", "").strip()
        if local and os.path.isfile(local):
            vgg_full = models.vgg19(weights=None)
            state = torch.load(local, map_location="cpu", weights_only=True)
            vgg_full.load_state_dict(state)
            vgg = vgg_full.features.eval()
        else:
            try:
                w = models.VGG19_Weights.IMAGENET1K_V1
            except AttributeError:
                w = models.VGG19_Weights.DEFAULT
            vgg = models.vgg19(weights=w).features.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        ch = list(vgg.children())
        self.stages = nn.ModuleList(
            [
                nn.Sequential(*ch[0:2]),
                nn.Sequential(*ch[2:7]),
                nn.Sequential(*ch[7:12]),
                nn.Sequential(*ch[12:21]),
                nn.Sequential(*ch[21:30]),
            ]
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer(
            "layer_w",
            torch.tensor([1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]),
        )

    def _to_rgb(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == 1:
            return x.repeat(1, 3, 1, 1)
        if x.size(1) == 3:
            return x
        raise ValueError(f"感知损失需要 1 或 3 通道，当前 {x.size(1)}")

    def _features(self, x_norm: torch.Tensor) -> list[torch.Tensor]:
        feats: list[torch.Tensor] = []
        cur = x_norm
        for stage in self.stages:
            cur = stage(cur)
            feats.append(cur)
        return feats

    def forward(self, pred: torch.Tensor, target: torch.Tensor, data_range: float) -> torch.Tensor:
        dr = float(data_range)
        x = self._to_rgb(pred) / dr
        y = self._to_rgb(target) / dr
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        with torch.no_grad():
            t_feats = self._features(y)
        p_feats = self._features(x)
        loss = pred.new_tensor(0.0)
        for i, w in enumerate(self.layer_w.tolist()):
            loss = loss + w * F.l1_loss(p_feats[i], t_feats[i])
        return loss


def local_pool(image, size):
    # Same adaptive-average bins, without the nondeterministic CUDA backward.
    height, width = image.shape[-2:]
    if height % size == 0 and width % size == 0:
        return F.avg_pool2d(image, (height // size, width // size))
    rows = []
    for y in range(size):
        top, bottom = y * height // size, ((y + 1) * height + size - 1) // size
        cells = []
        for x in range(size):
            left, right = x * width // size, ((x + 1) * width + size - 1) // size
            cells.append(image[..., top:bottom, left:right].mean((-2, -1)))
        rows.append(torch.stack(cells, -1))
    return torch.stack(rows, -2)


def proxies(image):
    mean = image.mean((-2, -1), keepdim=True)
    scale = ((image-mean).square().mean((-2, -1), keepdim=True) + 1e-4).sqrt()
    relative = (image-mean) / scale
    thresholds = image.new_tensor((-.5, 0., .5)).view(1, 3, 1, 1)
    q = torch.sigmoid((relative - thresholds) / .25)
    return torch.cat((q, 1 - q), 1)


def valid_open(q, radius):
    kernel = 2 * radius + 1
    eroded = -F.max_pool2d(-q, kernel, stride=1)
    return F.max_pool2d(eroded, kernel, stride=1)


def granulometry_loss(pred, target, pitch):
    if not torch.isfinite(pitch).all() or not (pitch > 0).all():
        raise ValueError("Positive declared sampling pitch required")
    q, reference = proxies(pred), proxies(target.detach())
    values = []
    for i, spacing in enumerate(pitch.detach().cpu().tolist()):
        for distance in (.15, .3, .6):
            radius = max(1, round(distance / spacing))
            if 4 * radius >= min(q.shape[-2:]) - 3:
                raise ValueError("Opening radius exceeds image support")
            a, b = valid_open(q[i:i+1], radius), valid_open(reference[i:i+1], radius)
            values.append(F.l1_loss(local_pool(a, 4), local_pool(b, 4)))
    return torch.stack(values).mean()


def patch_spectrum(image):
    patches = image.unfold(2, 64, 32).unfold(3, 64, 32)
    patches = patches - patches.mean((-2, -1), keepdim=True)
    window = torch.hann_window(64, periodic=False, dtype=image.dtype, device=image.device)
    patches = patches * window[:, None] * window[None, :]
    return torch.log1p(torch.fft.rfft2(patches, norm="ortho").abs())


def patch_spectrum_loss(pred, target):
    return F.l1_loss(patch_spectrum(pred), patch_spectrum(target.detach()))
