"""Masked Autoencoder: encoder sees visible patches only, light decoder reconstructs masked ones."""
import torch
import torch.nn as nn
from ..models.vit import Block, sincos_pos_embed


def patchify(imgs, p):
    b, c, h, w = imgs.shape
    x = imgs.reshape(b, c, h // p, p, w // p, p)
    return torch.einsum("nchpwq->nhwpqc", x).reshape(b, (h // p) * (w // p), p * p * c)


def unpatchify(x, p, c=3):
    b, n, _ = x.shape
    g = int(n ** 0.5)
    x = x.reshape(b, g, g, p, p, c)
    return torch.einsum("nhwpqc->nchpwq", x).reshape(b, c, g * p, g * p)


def random_masking(b, n, mask_ratio, device):
    """Returns keep indices (B, K), restore indices (B, N), binary mask (B, N) with 1 = masked."""
    k = int(n * (1 - mask_ratio))
    noise = torch.rand(b, n, device=device)
    shuffle = noise.argsort(1)
    restore = shuffle.argsort(1)
    mask = torch.ones(b, n, device=device)
    mask[:, :k] = 0
    return shuffle[:, :k], restore, torch.gather(mask, 1, restore)


class MAE(nn.Module):
    def __init__(self, encoder, mask_ratio=0.75, dec_dim=128, dec_depth=4, dec_heads=4, norm_pix=True):
        super().__init__()
        self.encoder, self.mask_ratio, self.norm_pix = encoder, mask_ratio, norm_pix
        p, n = encoder.patch, encoder.patch_embed.num_patches
        self.enc_to_dec = nn.Linear(encoder.dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.register_buffer("dec_pos", sincos_pos_embed(dec_dim, encoder.patch_embed.grid)[None], persistent=False)
        self.decoder = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.pred = nn.Linear(dec_dim, p * p * 3)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, imgs, mask_ratio=None):
        r = self.mask_ratio if mask_ratio is None else mask_ratio
        b, n = imgs.shape[0], self.encoder.patch_embed.num_patches
        keep, restore, mask = random_masking(b, n, r, imgs.device)
        lat = self.enc_to_dec(self.encoder.tokens(imgs, keep))
        cls, vis = lat[:, :1], lat[:, 1:]
        x = torch.cat([vis, self.mask_token.expand(b, n - vis.shape[1], -1)], 1)
        x = torch.gather(x, 1, restore[..., None].expand(-1, -1, x.shape[-1])) + self.dec_pos
        x = torch.cat([cls, x], 1)
        for blk in self.decoder:
            x = blk(x)
        pred = self.pred(self.dec_norm(x))[:, 1:]
        target = patchify(imgs, self.encoder.patch)
        if self.norm_pix:
            mu, var = target.mean(-1, keepdim=True), target.var(-1, keepdim=True)
            target = (target - mu) / (var + 1e-6) ** 0.5
        loss = ((pred - target) ** 2).mean(-1)
        loss = (loss * mask).sum() / mask.sum()  # masked patches only
        return loss, pred, mask
