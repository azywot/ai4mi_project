import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic convolutional building block
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        return self.dropout(self.act(self.bn(self.conv(x))))

# Downsampling block: conv(s) + optional pooling (or stride)
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, stride=2)  # halves spatial dims
    def forward(self, x):
        return self.conv(x)

# Upsampling + fusion block
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        # Add skip connection refinement
        self.skip_conv = ConvBlock(skip_ch, skip_ch, kernel_size=1, padding=0)
        self.conv = nn.Sequential(
            ConvBlock(in_ch + skip_ch, out_ch),
            ConvBlock(out_ch, out_ch)  # Add second conv for better feature extraction
        )
    
    def forward(self, x, skip):
        x = self.up(x)
        skip = self.skip_conv(skip)  # Refine skip connection
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# Patch embedding: convert spatial feature map to tokens
class PatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim, patch_size=1):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H/ps, W/ps)
        B, E, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, E), N = Hp * Wp
        return tokens, (Hp, Wp)

# Transformer Encoder
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(
                nn.TransformerEncoderLayer(
                    d_model=embed_dim,
                    nhead=num_heads,
                    dim_feedforward=int(embed_dim * mlp_ratio),
                    dropout=dropout,
                    batch_first=True,
                    activation='gelu'  # GELU is better than ReLU
                )
            )
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(embed_dim)
        
        # Add learnable positional encoding - properly initialized
        self.pos_embed = None
        self.embed_dim = embed_dim
        
    def forward(self, x, hw_shape=None):
        # x: (B, N, embed_dim)
        # Initialize positional embedding if needed
        if self.pos_embed is None or self.pos_embed.shape[1] != x.shape[1]:
            pos_embed = torch.zeros(1, x.shape[1], self.embed_dim, device=x.device, dtype=x.dtype)
            nn.init.trunc_normal_(pos_embed, std=0.02)
            self.pos_embed = nn.Parameter(pos_embed)
        
        x = x + self.pos_embed  # Add positional encoding
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

class TransUNet(nn.Module):
    def __init__(self, in_ch=1, num_classes=5, base_ch=32, embed_dim=128, 
                 depth=4, n_heads=8, patch_size=1, deep_supervision=False, **kwargs):
        super().__init__()
        # CNN Encoder (U-Net style)
        self.enc1 = ConvBlock(in_ch, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        # Downsampling
        self.down1 = Down(base_ch, base_ch)
        self.down2 = Down(base_ch * 2, base_ch * 2)
        self.down3 = Down(base_ch * 4, base_ch * 4)
        self.down4 = Down(base_ch * 8, base_ch * 8)

        # Patch embedding and transformer
        self.patch_embed = PatchEmbed(base_ch * 8, embed_dim, patch_size=patch_size)
        self.trans_enc = TransformerEncoder(embed_dim, depth, n_heads)

        # Project tokens back to feature map
        self.proj_back = nn.Linear(embed_dim, base_ch * 8)

        # U-Net decoder (upsample + skip fusion)
        self.up3 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up1 = Up(base_ch * 2, base_ch * 1, base_ch)

        self.final = nn.Conv2d(base_ch, num_classes, kernel_size=1)
        self.base_ch = base_ch
        self.deep_supervision = deep_supervision
        
        if deep_supervision:
            self.aux_head3 = nn.Conv2d(base_ch * 4, num_classes, kernel_size=1)
            self.aux_head2 = nn.Conv2d(base_ch * 2, num_classes, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        input_size = (H, W)

        # Encoder path - using Down blocks for downsampling
        e1 = self.enc1(x)           # resolution H, W
        e2 = self.enc2(self.down1(e1))  # resolution H/2, W/2
        e3 = self.enc3(self.down2(e2))  # resolution H/4, W/4
        e4 = self.enc4(self.down3(e3))  # resolution H/8, W/8

        # Tokenize e4 → tokens
        tokens, (Hp, Wp) = self.patch_embed(e4)
        tokens = self.trans_enc(tokens)  # (B, N, embed_dim)

        # Project back to feature map
        feat = self.proj_back(tokens)  # (B, N, base_ch*8)
        feat = feat.transpose(1, 2).reshape(B, self.base_ch * 8, Hp, Wp)

        # Decode: up + skip fusion
        d3 = self.up3(feat, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)

        logits = self.final(d1)
        # If necessary, upsample to input size
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision and self.training:
            aux3 = self.aux_head3(d3)
            aux2 = self.aux_head2(d2)
            aux3 = F.interpolate(aux3, size=input_size, mode='bilinear', align_corners=False)
            aux2 = F.interpolate(aux2, size=input_size, mode='bilinear', align_corners=False)
            return logits, aux3, aux2
        
        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'):
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        # Special initialization for final classification layer
        # Initialize biases to encourage all classes to be predicted
        if hasattr(self, 'final') and self.final.bias is not None:
            # Small negative bias for background, small positive for foreground classes
            with torch.no_grad():
                self.final.bias[0] = -0.5  # Background slightly discouraged
                self.final.bias[1:] = 0.1  # Foreground classes slightly encouraged
