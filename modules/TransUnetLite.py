import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic conv block with residual connection option
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1, stride=1, residual=False, dropout=0.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout2d(p=dropout) if dropout and dropout > 0.0 else nn.Identity()
        self.residual = residual
        # 1x1 conv for residual connection when channels differ
        self.shortcut = nn.Identity()
        if residual and in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)
    
    def forward(self, x):
        out = self.act(self.bn(self.conv(x)))
        out = self.drop(out)
        if self.residual:
            out = out + self.shortcut(x)
        return out


class AttentionGate(nn.Module):
    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int | None = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(1, skip_channels // 2)
        self.theta = nn.Conv2d(skip_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.phi = nn.Conv2d(gate_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        # Ensure spatial sizes match
        if skip.shape[2:] != gate.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode='bilinear', align_corners=False)
        theta_x = self.theta(skip)
        phi_g = self.phi(gate)
        attn = self.act(theta_x + phi_g)
        attn = self.sigmoid(self.psi(attn))
        return skip * attn


def build_sincos_2d_pos_embed(H: int, W: int, embed_dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # 2D sinusoidal positional encoding; supports any embed_dim, pads/truncates as needed
    if embed_dim <= 0:
        return torch.zeros((H * W, 0), device=device, dtype=dtype)
    num_feats = max(1, embed_dim // 4)
    y_embed = torch.arange(H, device=device, dtype=dtype).unsqueeze(1).repeat(1, W)
    x_embed = torch.arange(W, device=device, dtype=dtype).unsqueeze(0).repeat(H, 1)

    dim_t = torch.arange(num_feats, device=device, dtype=dtype)
    dim_t = 10000 ** (2 * (dim_t // 2) / float(num_feats))

    pos_y = y_embed.unsqueeze(-1) / dim_t  # H x W x C
    pos_x = x_embed.unsqueeze(-1) / dim_t  # H x W x C

    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # H x W x 2C
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # H x W x 2C

    pos = torch.cat((pos_y, pos_x), dim=-1)  # H x W x 4C
    total_dim = pos.shape[-1]
    if total_dim < embed_dim:
        pad = torch.zeros((H, W, embed_dim - total_dim), device=device, dtype=dtype)
        pos = torch.cat([pos, pad], dim=-1)
    elif total_dim > embed_dim:
        pos = pos[..., :embed_dim]
    pos = pos.view(H * W, embed_dim)
    return pos

# Deeper encoder with residual connections
class SlimEncoder(nn.Module):
    def __init__(self, in_ch, base_ch=24, dropout=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base_ch, residual=False, dropout=dropout)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, stride=2, residual=True, dropout=dropout)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, stride=2, residual=True, dropout=dropout)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, stride=2, residual=True, dropout=dropout)
    
    def forward(self, x):
        e1 = self.enc1(x)           # H, W
        e2 = self.enc2(e1)          # H/2, W/2
        e3 = self.enc3(e2)          # H/4, W/4
        e4 = self.enc4(e3)          # H/8, W/8
        return [e1, e2, e3, e4]

# Patch embedding for transformer
class SlimPatchEmbed(nn.Module):
    def __init__(self, in_ch, embed_dim=64, patch_size=2):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)
        B, E, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return tokens, (Hp, Wp)

# Transformer encoder with increased depth
class SlimTransformerEncoder(nn.Module):
    def __init__(self, dim, depth=4, num_heads=4, mlp_ratio=2.0, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=int(dim * mlp_ratio),
                dropout=dropout,
                batch_first=True,
                activation='gelu'
            ) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x

# Decoder + fusion with skip connections and deep supervision
class SlimDecoderFusion(nn.Module):
    def __init__(self, embed_dim, decoder_chs, skip_chs, num_classes, deep_supervision=False, dropout=0.1, use_attention_skip=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.use_attention_skip = use_attention_skip
        # Project token → feature map
        self.token_to_map = nn.Linear(embed_dim, decoder_chs[0])
        self.up_blocks = nn.ModuleList()
        self.conv_blocks = nn.ModuleList()
        self.skip_convs = nn.ModuleList()  # Skip connection refinement
        self.attn_gates = nn.ModuleList() if use_attention_skip else None
        
        for i in range(len(decoder_chs) - 1):
            in_ch = decoder_chs[i] + skip_chs[i]
            out_ch = decoder_chs[i + 1]
            # upsample via transpose convolution
            self.up_blocks.append(
                nn.ConvTranspose2d(decoder_chs[i], decoder_chs[i], kernel_size=2, stride=2)
            )
            # Skip connection refinement with 1x1 conv
            self.skip_convs.append(
                nn.Conv2d(skip_chs[i], skip_chs[i], kernel_size=1)
            )
            if self.use_attention_skip:
                self.attn_gates.append(AttentionGate(skip_channels=skip_chs[i], gate_channels=decoder_chs[i]))
            self.conv_blocks.append(ConvBlock(in_ch, out_ch, residual=True, dropout=dropout))
        
        self.final = nn.Conv2d(decoder_chs[-1], num_classes, kernel_size=1)
        
        # Deep supervision auxiliary heads
        if deep_supervision:
            self.aux_heads = nn.ModuleList([
                nn.Conv2d(decoder_chs[i+1], num_classes, kernel_size=1)
                for i in range(len(decoder_chs) - 2)  # Don't include final layer
            ])

    def forward(self, tokens, hw, skip_feats, input_size):
        Hp, Wp = hw
        B, N, E = tokens.shape
        # project tokens to decoder_chs[0] channels
        x = self.token_to_map(tokens)  # (B, N, dec_ch0)
        x = x.transpose(1, 2).reshape(B, -1, Hp, Wp)
        out = x
        aux_outputs = []
        
        for i, (up, skip_conv, conv) in enumerate(zip(self.up_blocks, self.skip_convs, self.conv_blocks)):
            out = up(out)
            skip = skip_conv(skip_feats[i])  # Refine skip connection
            if self.use_attention_skip:
                skip = self.attn_gates[i](skip, out)
            if out.shape[2:] != skip.shape[2:]:
                out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=False)
            out = torch.cat([out, skip], dim=1)
            out = conv(out)
            
            # Deep supervision: store intermediate outputs
            if self.deep_supervision and self.training and i < len(self.aux_heads):
                aux_logits = self.aux_heads[i](out)
                if aux_logits.shape[2:] != input_size:
                    aux_logits = F.interpolate(aux_logits, size=input_size, mode='bilinear', align_corners=False)
                aux_outputs.append(aux_logits)
        
        # final logits
        logits = self.final(out)
        # Upsample logits to input size if necessary
        if logits.shape[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        
        if self.deep_supervision and self.training and len(aux_outputs) > 0:
            return logits, *aux_outputs
        return logits

class TransUNetLite(nn.Module):
    def __init__(self, in_ch=1, num_classes=5,
                 base_ch=24, embed_dim=64, trans_depth=4, n_heads=4, deep_supervision=False,
                 dropout=0.1, trans_dropout=0.1, use_positional_encoding=True, use_attention_skip=True):
        super().__init__()
        self.use_positional_encoding = use_positional_encoding
        self.encoder = SlimEncoder(in_ch, base_ch=base_ch, dropout=dropout)
        # With 4 encoder stages: e1, e2, e3, e4
        # Skip connections from e3, e2, e1 (reverse order for decoder)
        skip1 = base_ch * 4  # from e3
        skip2 = base_ch * 2  # from e2
        skip3 = base_ch      # from e1
        deepest = base_ch * 8  # e4
        
        self.patch_embed = SlimPatchEmbed(deepest, embed_dim=embed_dim, patch_size=2)
        self.trans_enc = SlimTransformerEncoder(embed_dim, depth=trans_depth, num_heads=n_heads, dropout=trans_dropout)
        
        # Decoder channels: start from deepest, go to base
        decoder_chs = [deepest, base_ch * 4, base_ch * 2, base_ch]
        skip_chs = [skip1, skip2, skip3]
        self.decoder = SlimDecoderFusion(embed_dim, decoder_chs, skip_chs, num_classes, deep_supervision, dropout=dropout, use_attention_skip=use_attention_skip)

    def forward(self, x):
        B, C, H, W = x.shape
        input_size = (H, W)
        e1, e2, e3, e4 = self.encoder(x)
        tokens, hw = self.patch_embed(e4)
        if self.use_positional_encoding:
            Hp, Wp = hw
            pos = build_sincos_2d_pos_embed(Hp, Wp, tokens.size(-1), device=tokens.device, dtype=tokens.dtype)
            pos = pos.unsqueeze(0).expand(B, -1, -1)
            tokens = tokens + pos
        tokens = self.trans_enc(tokens)
        # Skip connections in reverse order for decoder
        skip_feats = [e3, e2, e1]
        logits = self.decoder(tokens, hw, skip_feats, input_size)
        return logits

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if hasattr(m, 'weight'):
                    nn.init.ones_(m.weight)
                if hasattr(m, 'bias'):
                    nn.init.zeros_(m.bias)
