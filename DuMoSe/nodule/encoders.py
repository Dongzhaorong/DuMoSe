"""3D ResNet morphology encoder and 3D ViT visual-semantic encoder."""
import torch
from torch import nn
from torch.nn import functional as F

def _make_layer3d(block, in_planes, planes, blocks, stride):
    layers = [block(in_planes, planes, stride=stride)]
    for _ in range(1, blocks):
        layers.append(block(planes * block.expansion, planes, stride=1))
    return nn.Sequential(*layers), planes * block.expansion

class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out

class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out = self.relu(out + identity)
        return out

class ResNet3DEncoder(nn.Module):
    """
    A 3D ResNet encoder that outputs:
      - vec: global pooled feature (B, C)
      - fmap (optional): last feature map (B, C, D', H', W')
    Note: vec dim equals fmap channels (C), matching DenseNet3DEncoder behavior.
    """

    def __init__(self, depth=101, base_channels=64, in_channels=1):
        super().__init__()
        depth = int(depth)
        if depth == 18:
            block, layers = BasicBlock3D, (2, 2, 2, 2)
        elif depth == 34:
            block, layers = BasicBlock3D, (3, 4, 6, 3)
        elif depth == 50:
            block, layers = Bottleneck3D, (3, 4, 6, 3)
        elif depth == 101:
            block, layers = Bottleneck3D, (3, 4, 23, 3)
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        in_planes = base_channels
        self.layer1, in_planes = _make_layer3d(block, in_planes, base_channels, layers[0], stride=1)
        self.layer2, in_planes = _make_layer3d(block, in_planes, base_channels * 2, layers[1], stride=2)
        self.layer3, in_planes = _make_layer3d(block, in_planes, base_channels * 4, layers[2], stride=2)
        self.layer4, in_planes = _make_layer3d(block, in_planes, base_channels * 8, layers[3], stride=2)

        self.out_channels = int(in_planes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x, return_map: bool = False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        fmap = self.layer4(x)
        vec = F.adaptive_avg_pool3d(fmap, (1, 1, 1)).flatten(1)
        if return_map:
            return vec, fmap
        return vec

class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Identity()  # keep minimal; no stochastic depth
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        # x: [B,N,C]
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class ViT3DEncoder(nn.Module):
    """
    Minimal ViT3D:
    - patchify by Conv3d(kernel=patch, stride=patch)
    - add cls token + learnable pos embedding
    - Transformer encoder blocks
    - output cls token vector [B, dim]
    """

    def __init__(self,
                 vol_size: int = 48,
                 patch_size: int = 8,
                 in_chans: int = 1,
                 dim: int = 256,
                 depth: int = 6,
                 num_heads: int = 8,
                 mlp_ratio: float = 4.0,
                 drop: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        if vol_size % patch_size != 0:
            raise ValueError(f"vol_size ({vol_size}) must be divisible by patch_size ({patch_size})")
        self.vol_size = int(vol_size)
        self.patch_size = int(patch_size)
        self.dim = int(dim)

        self.patch_embed = nn.Conv3d(in_chans, dim, kernel_size=patch_size, stride=patch_size, bias=True)

        grid = vol_size // patch_size
        num_patches = grid * grid * grid
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_drop = nn.Dropout(drop)

        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop=drop, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x, return_tokens: bool = False):
        # x: [B,1,D,H,W]
        x = self.patch_embed(x)  # [B,dim,D',H',W']
        x = x.flatten(2).transpose(1, 2)  # [B,N,dim]
        B, N, C = x.shape

        cls = self.cls_token.expand(B, -1, -1)  # [B,1,dim]
        x = torch.cat([cls, x], dim=1)  # [B,1+N,dim]
        x = x + self.pos_embed[:, : (N + 1), :]
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        if return_tokens:
            return x[:, 0], x[:, 1:]  # (cls, patch_tokens)
        return x[:, 0]  # cls token

class ViT3DSemanticTower(nn.Module):
    def __init__(self, vit3d: ViT3DEncoder, embed_dim: int = 512,
                 use_mlp: bool = True, hidden_dim: int = 0):
        super().__init__()
        self.vit = vit3d
        in_dim = vit3d.dim
        if use_mlp:
            h = in_dim if hidden_dim <= 0 else hidden_dim
            self.proj = nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Linear(h, embed_dim),
            )
        else:
            self.proj = nn.Linear(in_dim, embed_dim)

    def forward(self, vols):
        cls, toks = self.vit(vols, return_tokens=True)  # [B,dim], [B,N,dim]
        cls = F.normalize(self.proj(cls).float(), dim=-1)
        toks = F.normalize(self.proj(toks).float(), dim=-1)
        return cls, toks
