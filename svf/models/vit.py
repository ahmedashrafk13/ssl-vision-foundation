"""Vision Transformer: patch embedding + pre-norm Transformer encoder blocks."""
import torch
import torch.nn as nn


def sincos_pos_embed(dim, grid):
    """Fixed 2D sin-cos positional embedding, shape (grid*grid, dim)."""
    y, x = torch.meshgrid(torch.arange(grid, dtype=torch.float32),
                          torch.arange(grid, dtype=torch.float32), indexing="ij")
    omega = 1.0 / 10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4))
    oy, ox = torch.outer(y.flatten(), omega), torch.outer(x.flatten(), omega)
    return torch.cat([oy.sin(), oy.cos(), ox.sin(), ox.cos()], 1)


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch, dim):
        super().__init__()
        self.grid = img_size // patch
        self.num_patches = self.grid ** 2
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # B, N, D


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, drop_path=0.0):
        super().__init__()
        self.n1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = (torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path).to(x.dtype)
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        h = self.n1(x)
        x = x + self._dp(self.attn(h, h, h, need_weights=False)[0])
        return x + self._dp(self.mlp(self.n2(x)))


class ViT(nn.Module):
    def __init__(self, img_size=224, patch=16, dim=192, depth=12, heads=3, drop_path=0.0):
        super().__init__()
        self.patch, self.dim = patch, dim
        self.patch_embed = PatchEmbed(img_size, patch, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos_embed", sincos_pos_embed(dim, self.patch_embed.grid)[None], persistent=False)
        self.blocks = nn.ModuleList([Block(dim, heads, drop_path=drop_path * i / max(1, depth - 1))
                                     for i in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out_dim = dim
        self.channels = [dim]
        nn.init.normal_(self.cls_token, std=0.02)

    def tokens(self, x, keep_idx=None):
        """CLS + patch tokens; if keep_idx (B, K) is given only those patches are encoded (MAE)."""
        t = self.patch_embed(x) + self.pos_embed
        if keep_idx is not None:
            t = torch.gather(t, 1, keep_idx[..., None].expand(-1, -1, t.shape[-1]))
        t = torch.cat([self.cls_token.expand(t.shape[0], -1, -1), t], 1)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    def forward_features(self, x):
        """Patch tokens reshaped to a (B, D, H/p, W/p) map for dense heads."""
        t = self.tokens(x)[:, 1:]
        g = self.patch_embed.grid
        return {"c5": t.transpose(1, 2).reshape(x.shape[0], self.dim, g, g)}

    def forward(self, x):
        return self.tokens(x)[:, 0]


def vit_tiny(**kw):
    return ViT(dim=192, depth=12, heads=3, **kw)


def vit_small(**kw):
    return ViT(dim=384, depth=12, heads=6, **kw)
