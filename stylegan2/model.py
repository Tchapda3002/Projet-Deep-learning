"""
StyleGAN2 — Architecture complète pour images 256x256.

Innovations clés vs DCGAN / ProGAN :
  1. Mapping Network  : z -> w (espace de style intermédiaire)
  2. Weight Demodulation : remplace AdaIN + BatchNorm, plus stable
  3. Input constant appris : le générateur part d'un tenseur fixe 4x4
  4. R1 Regularization : pénalise le gradient du discriminateur sur les vraies images
  5. EMA Generator : moyenne exponentielle des poids -> images plus nettes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ─── Mapping Network ──────────────────────────────────────────────────────────

class MappingNetwork(nn.Module):
    """
    Transforme le vecteur de bruit z en vecteur de style w.

    Pourquoi ? L'espace z suit une distribution gaussienne — certaines
    combinaisons d'attributs sont impossibles. L'espace w appris est
    "désentrelacé" : chaque dimension contrôle un attribut indépendant.

    Architecture : 8 couches linéaires avec LeakyReLU.
    """
    def __init__(self, z_dim=512, w_dim=512, n_layers=8):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers += [nn.Linear(z_dim, w_dim), nn.LeakyReLU(0.2)]
            z_dim = w_dim
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        # Normalisation du vecteur z avant le mapping
        z = F.normalize(z, dim=1)
        return self.net(z)


# ─── Weight Demodulation ──────────────────────────────────────────────────────

class ModulatedConv2d(nn.Module):
    """
    Convolution modulée par le vecteur de style w.

    Étapes :
      1. Modulation   : scale les poids avec w (via couche affine)
      2. Démodulation : normalise les poids pour éviter l'explosion
      3. Convolution  : applique les poids modulés/démodulés

    C'est le coeur de StyleGAN2 — remplace complètement BatchNorm.
    """
    def __init__(self, in_ch, out_ch, kernel, w_dim=512, demod=True, upsample=False):
        super().__init__()
        self.out_ch   = out_ch
        self.kernel   = kernel
        self.demod    = demod
        self.upsample = upsample
        self.padding  = kernel // 2

        # Poids de la convolution
        self.weight = nn.Parameter(
            torch.randn(1, out_ch, in_ch, kernel, kernel)
        )
        # Couche affine : w -> scale par canal d'entrée
        self.affine = nn.Linear(w_dim, in_ch)
        nn.init.ones_(self.affine.bias)

        self.scale = 1 / np.sqrt(in_ch * kernel * kernel)

    def forward(self, x, w):
        B, C, H, W = x.shape

        # 1. Modulation
        style = self.affine(w).view(B, 1, C, 1, 1)           # (B, 1, C, 1, 1)
        weight = self.weight * self.scale * style              # (B, out, in, k, k)

        # 2. Démodulation
        if self.demod:
            sigma = weight.pow(2).sum(dim=[2, 3, 4], keepdim=True).add(1e-8).sqrt()
            weight = weight / sigma

        # 3. Convolution groupée par batch
        x = x.contiguous().reshape(1, B * C, H, W)
        weight = weight.contiguous().reshape(B * self.out_ch, C, self.kernel, self.kernel)

        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        x = F.conv2d(x, weight, padding=self.padding, groups=B)
        return x.reshape(B, self.out_ch, x.size(2), x.size(3))


# ─── Bruit injecté ────────────────────────────────────────────────────────────

class NoiseInjection(nn.Module):
    """
    Ajoute du bruit aléatoire après chaque convolution.
    Contrôle les détails fins (pores, cheveux) sans affecter la structure globale.
    """
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        return x + self.weight * noise


# ─── Bloc Générateur ──────────────────────────────────────────────────────────

class StyleBlock(nn.Module):
    """
    Un bloc du générateur StyleGAN2 :
      ModulatedConv -> NoiseInjection -> LeakyReLU  (x2)
    """
    def __init__(self, in_ch, out_ch, w_dim=512, upsample=False):
        super().__init__()
        self.conv1  = ModulatedConv2d(in_ch,  out_ch, 3, w_dim, upsample=upsample)
        self.conv2  = ModulatedConv2d(out_ch, out_ch, 3, w_dim)
        self.noise1 = NoiseInjection()
        self.noise2 = NoiseInjection()
        self.act    = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x, w):
        x = self.act(self.noise1(self.conv1(x, w)))
        x = self.act(self.noise2(self.conv2(x, w)))
        return x


# ─── Générateur ───────────────────────────────────────────────────────────────

class StyleGenerator(nn.Module):
    """
    Génère une image 256x256 depuis un vecteur de style w.

    Pipeline :
      z -> MappingNetwork -> w
      Input constant 4x4 -> StyleBlock (4x4) -> StyleBlock (8x8) -> ...
      -> StyleBlock (256x256) -> ToRGB -> image

    L'input constant est un tenseur appris — le générateur ne dépend
    pas de z directement, seulement via w qui module les poids.

    synthesis(w) permet d'injecter w directement (truncation trick).
    """
    def __init__(self, z_dim=512, w_dim=512, channels=3):
        super().__init__()

        self.mapping = MappingNetwork(z_dim, w_dim)

        # Input constant appris (4x4x512)
        self.const = nn.Parameter(torch.randn(1, 512, 4, 4))

        # Filtres par résolution : 4, 8, 16, 32, 64, 128, 256
        feats = [512, 512, 512, 256, 128, 64, 32]

        # Bloc initial (4x4, pas d'upsample)
        self.init_block = StyleBlock(512, 512, w_dim, upsample=False)

        # Blocs progressifs : chaque bloc double la résolution
        # 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 256x256
        self.blocks = nn.ModuleList([
            StyleBlock(feats[i], feats[i + 1], w_dim, upsample=True)
            for i in range(len(feats) - 1)
        ])

        # Couche ToRGB finale
        self.to_rgb = ModulatedConv2d(feats[-1], channels, 1, w_dim, demod=False)
        self.bias   = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def synthesis(self, w):
        """Partie synthèse uniquement — prend w directement (pour truncation trick)."""
        x = self.const.expand(w.size(0), -1, -1, -1)
        x = self.init_block(x, w)
        for block in self.blocks:
            x = block(x, w)
        rgb = self.to_rgb(x, w) + self.bias
        return torch.tanh(rgb)

    def forward(self, z):
        w = self.mapping(z)
        return self.synthesis(w)


# ─── Discriminateur ───────────────────────────────────────────────────────────

class StyleDiscriminator(nn.Module):
    """
    Discriminateur résiduel avec Spectral Normalization.

    La Spectral Normalization contraint le discriminateur à être
    Lipschitz-1 — empêche les sorties de croître de façon incontrôlée
    et stabilise l'entraînement sans avoir besoin de R1 regularization.
    """
    def __init__(self, channels=3):
        super().__init__()
        from torch.nn.utils import spectral_norm

        feats = [32, 64, 128, 256, 512, 512, 512]

        self.from_rgb = nn.Sequential(
            spectral_norm(nn.Conv2d(channels, feats[0], 1)),
            nn.LeakyReLU(0.2)
        )

        blocks = []
        for i in range(len(feats) - 1):
            blocks += [
                spectral_norm(nn.Conv2d(feats[i], feats[i], 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                spectral_norm(nn.Conv2d(feats[i], feats[i + 1], 3, padding=1)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.AvgPool2d(2),
            ]
        self.blocks = nn.Sequential(*blocks)

        # Bloc final : MinibatchStd + FC
        # Après 6 AvgPool2d sur entrée 256x256 : 256/64 = 4x4
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(feats[-1] + 1, feats[-1], 3, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            spectral_norm(nn.Linear(feats[-1] * 4 * 4, feats[-1])),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Linear(feats[-1], 1)),
        )

    def _minibatch_std(self, x):
        std = x.std(dim=0, keepdim=True).mean().expand(x.size(0), 1, x.size(2), x.size(3))
        return torch.cat([x, std], dim=1)

    def forward(self, img):
        x = self.from_rgb(img)
        x = self.blocks(x)
        x = self._minibatch_std(x)
        return self.final(x).view(-1)


def build_stylegan2(z_dim=512, w_dim=512, channels=3, device='cpu'):
    """Construit et retourne (G, D) prêts à l'entraînement."""
    G = StyleGenerator(z_dim, w_dim, channels).to(device)
    D = StyleDiscriminator(channels).to(device)
    return G, D
