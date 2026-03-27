import sys, os, torch, torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from model import build_stylegan2
from utils.loader import FFHQDataset
from utils.visualizer import save_snapshot, plot_losses


def r1_regularization(D, real_imgs, gamma):
    real_imgs = real_imgs.detach().requires_grad_(True)
    real_score = D(real_imgs).sum()
    grad = torch.autograd.grad(real_score, real_imgs, create_graph=True)[0]
    return (gamma / 2) * grad.pow(2).sum([1, 2, 3]).mean()


def train():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU : {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")

    print(f"MODE   : {cfg.MODE}")
    print(f"Device : {device}")

    os.makedirs(cfg.CHECKPOINTS_DIR,    exist_ok=True)
    os.makedirs(cfg.GENERATED_IMGS_DIR, exist_ok=True)
    os.makedirs(cfg.METRICS_DIR,        exist_ok=True)

    dataset    = FFHQDataset(cfg.DATA_PATH, cfg.IMG_SIZE)
    dataloader = DataLoader(dataset, batch_size=cfg.BATCH_SIZE,
                            shuffle=True, num_workers=cfg.NUM_WORKERS)
    print(f"Dataset : {len(dataset)} images | {len(dataloader)} batches/epoch")

    G, D = build_stylegan2(cfg.Z_DIM, cfg.W_DIM, cfg.CHANNELS, device)
    optimizer_G = optim.Adam(G.parameters(), lr=cfg.LR_G, betas=(cfg.BETA1, cfg.BETA2))
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.LR_D, betas=(cfg.BETA1, cfg.BETA2))

    z_fixed   = torch.randn(16, cfg.Z_DIM).to(device)
    losses_G  = []; losses_D = []; w_samples = []
    batch_count = 0

    print(f"Entrainement — {cfg.NUM_EPOCHS} epochs")
    print("-" * 55)

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        G.train(); D.train()
        sum_G = 0; sum_D = 0

        for real_imgs in dataloader:
            real_imgs = real_imgs.to(device)
            n = real_imgs.size(0)
            batch_count += 1

            optimizer_D.zero_grad()
            z = torch.randn(n, cfg.Z_DIM).to(device)
            fake_imgs = G(z)
            loss_D = (F.softplus(-D(real_imgs)).mean() +
                      F.softplus( D(fake_imgs.detach())).mean())
            if batch_count % cfg.R1_EVERY == 0:
                loss_D = loss_D + r1_regularization(D, real_imgs.clone(), cfg.R1_GAMMA)
            loss_D.backward()
            optimizer_D.step()

            optimizer_G.zero_grad()
            z = torch.randn(n, cfg.Z_DIM).to(device)
            fake_imgs = G(z)
            loss_G = F.softplus(-D(fake_imgs)).mean()
            loss_G.backward()
            optimizer_G.step()

            sum_G += loss_G.item()
            sum_D += loss_D.item()

            if batch_count % 10 == 0:
                with torch.no_grad():
                    w = G.mapping(z[:4])
                    w_samples.append(w.cpu())

        avg_G = sum_G / len(dataloader)
        avg_D = sum_D / len(dataloader)
        losses_G.append(avg_G)
        losses_D.append(avg_D)

        if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
            save_snapshot(G, z_fixed, epoch, cfg.GENERATED_IMGS_DIR, cfg.Z_DIM, device)
            ckpt = {
                "G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
                "optimizer_G": optimizer_G.state_dict(), "optimizer_D": optimizer_D.state_dict(),
                "epoch": epoch, "losses_G": losses_G, "losses_D": losses_D,
                "w_samples": torch.cat(w_samples, dim=0) if w_samples else torch.tensor([]),
                "config": {"z_dim": cfg.Z_DIM, "w_dim": cfg.W_DIM, "img_size": cfg.IMG_SIZE},
            }
            torch.save(ckpt, os.path.join(cfg.CHECKPOINTS_DIR, f"stylegan2_ep{epoch:04d}.pt"))
            print(f"Epoch [{epoch:3d}/{cfg.NUM_EPOCHS}] G={avg_G:.4f} D={avg_D:.4f} | saved")
        else:
            print(f"Epoch [{epoch:3d}/{cfg.NUM_EPOCHS}] G={avg_G:.4f} D={avg_D:.4f}")

    plot_losses(losses_G, losses_D, cfg.METRICS_DIR)
    w_tensor = torch.cat(w_samples, dim=0) if w_samples else torch.tensor([])
    torch.save({
        "G_state_dict": G.state_dict(), "D_state_dict": D.state_dict(),
        "epoch": cfg.NUM_EPOCHS, "losses_G": losses_G, "losses_D": losses_D,
        "w_samples": w_tensor,
        "config": {"z_dim": cfg.Z_DIM, "w_dim": cfg.W_DIM, "img_size": cfg.IMG_SIZE, "mode": cfg.MODE},
    }, os.path.join(cfg.CHECKPOINTS_DIR, "stylegan2_final.pt"))
    print("-" * 55)
    print("Termine.")


if __name__ == "__main__":
    train()