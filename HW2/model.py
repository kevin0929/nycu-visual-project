import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------

class PositionEmbeddingSine(nn.Module):
    """2D sinusoidal positional encoding."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x, mask):
        # x: (B, C, H, W), mask: (B, H, W) - True where padded
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack([pos_x[:, :, :, 0::2].sin(),
                              pos_x[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos_y = torch.stack([pos_y[:, :, :, 0::2].sin(),
                              pos_y[:, :, :, 1::2].cos()], dim=4).flatten(3)
        pos = torch.cat([pos_y, pos_x], dim=3).permute(0, 3, 1, 2)
        return pos


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class BackboneWithFPN(nn.Module):
    """ResNet-50 backbone, returns last feature map + mask."""

    def __init__(self, pretrained=True, train_backbone=True):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet50(weights=weights)

        # Remove avgpool and fc
        self.body = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4,
        )
        self.num_channels = 2048

        if not train_backbone:
            for p in self.body.parameters():
                p.requires_grad_(False)

    def forward(self, x, mask):
        feat = self.body(x)
        # Downsample mask to match feature map size
        mask_down = F.interpolate(mask.unsqueeze(1).float(),
                                  size=feat.shape[-2:]).bool().squeeze(1)
        return feat, mask_down


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

class DETR_Transformer(nn.Module):
    """Standard Transformer encoder-decoder for DETR."""

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers,
                                              enable_nested_tensor=False)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)

        self.d_model = d_model

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        src:         (B, C, H, W)
        mask:        (B, H, W) — True = padded
        query_embed: (num_queries, d_model)
        pos_embed:   (B, d_model, H, W)
        """
        B, C, H, W = src.shape
        # Flatten spatial dims -> (B, HW, C)  [batch_first=True]
        src = src.flatten(2).permute(0, 2, 1)
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)
        mask_flat = mask.flatten(1)  # (B, HW)

        # Encoder
        memory = self.encoder(src + pos_embed, src_key_padding_mask=mask_flat)

        # Decoder: tgt shape (B, num_queries, d_model)
        tgt = torch.zeros(B, query_embed.shape[0], self.d_model, device=src.device)
        query_pos = query_embed.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(tgt + query_pos, memory,
                          memory_key_padding_mask=mask_flat,
                          tgt_key_padding_mask=None)
        # hs: (B, num_queries, d_model) — already correct for batch_first
        return hs, memory.permute(0, 2, 1).view(B, C, H, W)


# ---------------------------------------------------------------------------
# Prediction Heads (MLP)
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out)
                                     for d_in, d_out in zip(dims[:-1], dims[1:])])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


# ---------------------------------------------------------------------------
# Full DETR Model
# ---------------------------------------------------------------------------

class DETR(nn.Module):
    """
    DETR: End-to-End Object Detection with Transformers.
    Backbone: ResNet-50 (frozen BN by default).
    """

    def __init__(self,
                 num_classes=10,
                 num_queries=100,
                 d_model=256,
                 nhead=8,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dim_feedforward=2048,
                 dropout=0.1,
                 pretrained_backbone=True,
                 train_backbone=True):
        super().__init__()

        self.backbone = BackboneWithFPN(pretrained=pretrained_backbone,
                                        train_backbone=train_backbone)
        # Input projection: 2048 -> d_model
        self.input_proj = nn.Conv2d(self.backbone.num_channels, d_model, kernel_size=1)

        self.transformer = DETR_Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        # Object queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Positional encoding
        self.pos_encoder = PositionEmbeddingSine(d_model // 2)

        # Prediction heads
        # +1 for "no object" class
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)

        self.num_queries = num_queries
        self.num_classes = num_classes

    def forward(self, samples):
        """
        samples: NestedTensor with .tensors (B,3,H,W) and .mask (B,H,W)
        """
        tensors, mask = samples.tensors, samples.mask

        # Backbone
        feat, mask_down = self.backbone(tensors, mask)

        # Project features
        src = self.input_proj(feat)

        # Positional encoding
        pos = self.pos_encoder(src, mask_down)

        # Transformer
        hs, _ = self.transformer(src, mask_down, self.query_embed.weight, pos)

        # Predictions
        outputs_class = self.class_embed(hs)
        outputs_coord = self.bbox_embed(hs).sigmoid()

        return {"pred_logits": outputs_class, "pred_boxes": outputs_coord}


# ---------------------------------------------------------------------------
# NestedTensor helper
# ---------------------------------------------------------------------------

class NestedTensor:
    """Wraps a batch of images with their padding masks."""

    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        return NestedTensor(self.tensors.to(device), self.mask.to(device))


def nested_tensor_from_tensor_list(tensor_list):
    """Pad images to the same size and create a NestedTensor."""
    max_h = max(img.shape[1] for img in tensor_list)
    max_w = max(img.shape[2] for img in tensor_list)

    batch_size = len(tensor_list)
    batched = torch.zeros(batch_size, 3, max_h, max_w,
                          dtype=tensor_list[0].dtype,
                          device=tensor_list[0].device)
    mask = torch.ones(batch_size, max_h, max_w, dtype=torch.bool,
                      device=tensor_list[0].device)

    for i, img in enumerate(tensor_list):
        c, h, w = img.shape
        batched[i, :c, :h, :w].copy_(img)
        mask[i, :h, :w] = False  # False = valid pixel

    return NestedTensor(batched, mask)


def build_model(cfg):
    """Build DETR model from config dict."""
    return DETR(
        num_classes=cfg.get("num_classes", 10),
        num_queries=cfg.get("num_queries", 100),
        d_model=cfg.get("d_model", 256),
        nhead=cfg.get("nhead", 8),
        num_encoder_layers=cfg.get("num_encoder_layers", 6),
        num_decoder_layers=cfg.get("num_decoder_layers", 6),
        dim_feedforward=cfg.get("dim_feedforward", 2048),
        dropout=cfg.get("dropout", 0.1),
        pretrained_backbone=cfg.get("pretrained_backbone", True),
        train_backbone=cfg.get("train_backbone", True),
    )