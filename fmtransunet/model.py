"""TransUNet backbone used by FM-TransUNet."""

import torch
from torch import nn
from torch.nn import functional as F


class PlusDoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1,padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1,padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlusDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), PlusDoubleConv(in_ch, out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlusUp(nn.Module):
    """双线性 2× 上采样 + 与 skip 拼接 + DoubleConv（与旧版 bilinear=True 路径一致）。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = PlusDoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class BottleneckTransformerPreNorm(nn.Module):
    """t = t + Enc(LN(t))；2D 归一化坐标 → Linear 位置编码。"""

    def __init__(
        self,
        in_channels: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        dim_feedforward: int,
    ) -> None:
        super().__init__()
        self.proj_in = nn.Linear(in_channels, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.pos_proj = nn.Linear(2, d_model)
        self.proj_out = nn.Linear(d_model, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        t = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        t = self.proj_in(t)
        device = x.device
        ys = torch.linspace(-1.0, 1.0, steps=h, device=device)
        xs = torch.linspace(-1.0, 1.0, steps=w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        pos = torch.stack([gx, gy], dim=-1).reshape(1, h * w, 2).expand(b, -1, -1)
        t = t + self.pos_proj(pos)
        t = t + self.encoder(self.norm(t))
        t = self.proj_out(t)
        t = t.reshape(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return t


class TransUNetPlus(nn.Module):
    """
    encoder_channels 在 JSON 中作为 UNet features 列表（如 5 级 [64,…,1024]）；
    瓶颈通道 = features[-1] * 2 // (2 if bilinear else 1)。
    """

    def __init__(self, channels=(64, 128, 256, 512, 1024), embed_dim=768, depth=8,
                 heads=12, feedforward=3072, dropout=.1):
        super().__init__()
        features = list(channels)
        if len(features) < 2:
            raise ValueError("TransUNetPlus 需要 encoder_channels 至少 2 个尺度")

        in_ch, out_ch = 3, 1

        self.in_conv = PlusDoubleConv(in_ch, features[0])
        self.downs = nn.ModuleList()
        for idx in range(1, len(features)):
            self.downs.append(PlusDown(features[idx - 1], features[idx]))

        bottleneck_ch = features[-1]
        self.pre_bottleneck = PlusDoubleConv(features[-1], bottleneck_ch)

        dm = embed_dim
        dff = feedforward
        self.trans = BottleneckTransformerPreNorm(
            in_channels=bottleneck_ch,
            d_model=dm,
            nhead=heads,
            num_layers=depth,
            dropout=dropout,
            dim_feedforward=dff,
        )

        rev_feats = list(reversed(features))
        self.ups = nn.ModuleList()
        ch_list = [bottleneck_ch] + [f for f in rev_feats]
        for i in range(len(rev_feats)):
            inch = ch_list[i] + rev_feats[i]
            outch = rev_feats[i]
            self.ups.append(PlusUp(inch, outch))

        self.out_conv = nn.Conv2d(rev_feats[-1], out_ch, kernel_size=1)
        self._out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.in_conv(x)
        skips = [x0]
        cur = x0
        for down in self.downs:
            cur = down(cur)
            skips.append(cur)

        x = self.pre_bottleneck(cur)
        x = self.trans(x)
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip)
        return self._out_act(self.out_conv(x))
