import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrangea
import numbers


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, bias=True):
        super().__init__()
        self.body = WithBias_LayerNorm(dim) if bias else BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, 1, bias=bias)
        self.dw_conv = nn.Conv2d(
            hidden * 2, hidden * 2, 3, stride=1, padding=1,
            groups=hidden * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden, dim, 1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dw_conv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


# ---------------------------------------------------------------------------
# Multi-DConv Head Transposed Self-Attention (MDTA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=bias)
        self.qkv_dw = nn.Conv2d(
            dim * 3, dim * 3, 3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, 1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dw(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# Prompt Block
# ---------------------------------------------------------------------------

class PromptBlock(nn.Module):
    """Lightweight prompt generation block."""
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96):
        super().__init__()
        self.prompt_param = nn.Parameter(
            torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size)
        )
        self.linear_layer = nn.Linear(prompt_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        emb = x.mean(dim=(-2, -1))                          # (B, C)
        prompt_w = self.linear_layer(emb).softmax(dim=1)    # (B, prompt_len)
        # weighted sum over prompts
        prompt_param = self.prompt_param.expand(b, -1, -1, -1, -1)  # (B, L, D, S, S)
        prompt = torch.einsum("bl,bldhw->bdhw", prompt_w, prompt_param)
        prompt = F.interpolate(prompt, (h, w), mode="bilinear", align_corners=False)
        prompt = self.conv3x3(prompt)
        return prompt


# ---------------------------------------------------------------------------
# Transformer Block (with optional Prompt injection)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, use_prompt=False,
                 prompt_dim=None, prompt_len=5, prompt_size=96):
        super().__init__()
        self.use_prompt = use_prompt
        self.norm1 = LayerNorm(dim, bias=bias)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, bias=bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

        if use_prompt:
            assert prompt_dim is not None
            self.prompt_block = PromptBlock(
                prompt_dim=prompt_dim,
                prompt_len=prompt_len,
                prompt_size=prompt_size,
            )
            self.prompt_norm = LayerNorm(prompt_dim, bias=bias)

    def forward(self, x):
        if self.use_prompt:
            prompt = self.prompt_block(x)
            x = x + self.attn(self.norm1(x)) + self.prompt_norm(prompt)
        else:
            x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Down / Up sampling
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# ---------------------------------------------------------------------------
# PromptIR (U-Net with Transformer blocks + Prompt blocks at decoder)
# ---------------------------------------------------------------------------

class PromptIR(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=(4, 6, 6, 8),
        num_refinement_blocks=4,
        heads=(1, 2, 4, 8),
        ffn_expansion_factor=2.66,
        bias=False,
        # prompt settings
        prompt_dim=128,
        prompt_len=5,
        prompt_size=96,
        use_prompt=True,
    ):
        super().__init__()
        self.use_prompt = use_prompt

        # Patch embedding
        self.patch_embed = nn.Conv2d(inp_channels, dim, 3, 1, 1, bias=bias)

        # Encoder
        self.enc1 = nn.Sequential(*[
            TransformerBlock(dim, heads[0], ffn_expansion_factor, bias)
            for _ in range(num_blocks[0])
        ])
        self.down1_2 = Downsample(dim)

        self.enc2 = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[1], ffn_expansion_factor, bias)
            for _ in range(num_blocks[1])
        ])
        self.down2_3 = Downsample(dim * 2)

        self.enc3 = nn.Sequential(*[
            TransformerBlock(dim * 4, heads[2], ffn_expansion_factor, bias)
            for _ in range(num_blocks[2])
        ])
        self.down3_4 = Downsample(dim * 4)

        # Bottleneck
        self.latent = nn.Sequential(*[
            TransformerBlock(dim * 8, heads[3], ffn_expansion_factor, bias)
            for _ in range(num_blocks[3])
        ])

        # Decoder (with prompts)
        self.up4_3 = Upsample(dim * 8)
        self.reduce_ch3 = nn.Conv2d(dim * 8, dim * 4, 1, bias=bias)
        self.dec3 = nn.Sequential(*[
            TransformerBlock(
                dim * 4, heads[2], ffn_expansion_factor, bias,
                use_prompt=(use_prompt and i == 0),
                prompt_dim=dim * 4, prompt_len=prompt_len, prompt_size=prompt_size
            )
            for i in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(dim * 4)
        self.reduce_ch2 = nn.Conv2d(dim * 4, dim * 2, 1, bias=bias)
        self.dec2 = nn.Sequential(*[
            TransformerBlock(
                dim * 2, heads[1], ffn_expansion_factor, bias,
                use_prompt=(use_prompt and i == 0),
                prompt_dim=dim * 2, prompt_len=prompt_len, prompt_size=prompt_size
            )
            for i in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(dim * 2)
        self.dec1 = nn.Sequential(*[
            TransformerBlock(
                dim * 2, heads[0], ffn_expansion_factor, bias,
                use_prompt=(use_prompt and i == 0),
                prompt_dim=dim * 2, prompt_len=prompt_len, prompt_size=prompt_size
            )
            for i in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(dim * 2, heads[0], ffn_expansion_factor, bias)
            for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(dim * 2, out_channels, 3, 1, 1, bias=bias)

    def forward(self, inp):
        # Encoder
        x = self.patch_embed(inp)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1_2(e1))
        e3 = self.enc3(self.down2_3(e2))

        # Bottleneck
        lat = self.latent(self.down3_4(e3))

        # Decoder
        d3 = self.up4_3(lat)
        d3 = self.dec3(self.reduce_ch3(torch.cat([d3, e3], dim=1)))

        d2 = self.up3_2(d3)
        d2 = self.dec2(self.reduce_ch2(torch.cat([d2, e2], dim=1)))

        d1 = self.up2_1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.refinement(d1)
        out = self.output(out) + inp
        return out