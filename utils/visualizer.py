"""
Visualisation : snapshots d'images générées et courbes de loss.
"""

import os
import torch
import matplotlib.pyplot as plt


def save_snapshot(G, z_fixed, epoch, output_dir, z_dim, device, n=16):
    """
    Génère n images avec le générateur et les sauvegarde en grille.

    Args:
        G          : générateur (en mode eval automatiquement)
        z_fixed    : vecteur de bruit fixe pour suivre la progression
        epoch      : numéro d'epoch (pour le nom du fichier)
        output_dir : dossier de sauvegarde
        z_dim      : dimension du vecteur latent
        device     : device torch
        n          : nombre d'images à générer (défaut 16)
    """
    G.eval()
    with torch.no_grad():
        imgs = G(z_fixed[:n]).cpu()
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)

    cols = 4
    rows = n // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    fig.suptitle(f'StyleGAN2 — Epoch {epoch}', fontsize=11)

    for i, ax in enumerate(axes.flat):
        ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis('off')

    plt.tight_layout()
    path = os.path.join(output_dir, f'epoch_{epoch:04d}.png')
    plt.savefig(path, dpi=80, bbox_inches='tight')
    plt.close()
    G.train()
    return path


def plot_losses(losses_G, losses_D, output_dir):
    """
    Trace et sauvegarde les courbes de loss G et D.

    Args:
        losses_G   : liste des losses du générateur
        losses_D   : liste des losses du discriminateur
        output_dir : dossier de sauvegarde
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(losses_G, label='Générateur',     color='royalblue')
    ax.plot(losses_D, label='Discriminateur', color='tomato')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Évolution des pertes — StyleGAN2')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, 'losses.png')
    plt.savefig(path, dpi=100)
    plt.close()
    return path


def show_grid(imgs_tensor, title='', n=16):
    """
    Affiche une grille d'images (tenseur normalisé [-1,1]).
    Utilisé directement dans les notebooks.
    """
    imgs = (imgs_tensor.cpu() * 0.5 + 0.5).clamp(0, 1)
    n    = min(n, len(imgs))  # ne pas dépasser le nombre d'images disponibles
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if title:
        fig.suptitle(title, fontsize=11)
    axes_flat = axes.flat if hasattr(axes, 'flat') else [axes]
    for i, ax in enumerate(axes_flat):
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
        ax.axis('off')
    plt.tight_layout()
    plt.show()
