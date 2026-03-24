import os
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from nerf.utils.io import load_yaml, set_seed, ensure_dir, save_ckpt
from nerf.datasets.llff_seathru import SeaThruLLFFDataset
from nerf.datasets.ray_utils import sample_random_rays
from nerf.models.nerf_mlp import NeRFMLP
from nerf.models.medium import HomogeneousMedium, ConstantEnvLight
from nerf.render.volume_render import volume_render_rte_single_scatter
from nerf.pinn.rte_residual import rte_residual_along_rays
from nerf.utils.metrics import psnr, l1

def main(cfg_path: str):
    cfg = load_yaml(cfg_path)
    set_seed(cfg["train"]["seed"])
    device = cfg["train"]["device"]

    out_dir = os.path.join("outputs", os.path.basename(cfg["data"]["root"]).rstrip("/"))
    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "ckpts"))
    writer = SummaryWriter(log_dir=os.path.join(out_dir, "tb"))

    train_set = SeaThruLLFFDataset(
        root=cfg["data"]["root"],
        images_dir=cfg["data"]["images_dir"],
        downscale=cfg["data"]["downscale"],
        split="train",
        holdout=cfg["data"]["holdout"],
        device=device,
    )
    test_set = SeaThruLLFFDataset(
        root=cfg["data"]["root"],
        images_dir=cfg["data"]["images_dir"],
        downscale=cfg["data"]["downscale"],
        split="test",
        holdout=cfg["data"]["holdout"],
        device=device,
    )

    nerf = NeRFMLP(
        hidden=cfg["model"]["nerf"]["hidden"],
        depth=cfg["model"]["nerf"]["depth"],
        skips=tuple(cfg["model"]["nerf"]["skips"]),
        view_dependent=cfg["model"]["nerf"]["view_dependent"],
    ).to(device)

    medium = HomogeneousMedium(
        init_sigma_t=cfg["model"]["medium"]["init_sigma_t"],
        init_sigma_s=cfg["model"]["medium"]["init_sigma_s"],
        init_g=cfg["model"]["medium"]["init_g"],
    ).to(device)

    env = ConstantEnvLight(init_rgb=tuple(cfg["model"]["env"]["init_rgb"])).to(device)

    params = list(nerf.parameters()) + list(medium.parameters()) + list(env.parameters())
    optim = torch.optim.Adam(params, lr=cfg["train"]["lr"])

    iters = cfg["train"]["iters"]
    batch_rays = cfg["train"]["batch_rays"]

    near = cfg["render"]["near"]
    far = cfg["render"]["far"]
    n_samples = cfg["render"]["n_samples"]

    w_img = cfg["loss"]["w_img"]
    w_rte = cfg["loss"]["w_rte"]
    detach_scatter = cfg["loss"]["rte"]["detach_scatter"]

    def lr_schedule(step):
        warm = cfg["train"]["warmup"]
        if step < warm:
            return cfg["train"]["lr"] * (step + 1) / warm
        return cfg["train"]["lr"]

    def eval_one():
        nerf.eval(); medium.eval(); env.eval()
        with torch.no_grad():
            batch = test_set[0]
            rgb = batch["rgb"]
            rays_o = batch["rays_o"].reshape(-1, 3)
            rays_d = batch["rays_d"].reshape(-1, 3)
            pred, J, _ = volume_render_rte_single_scatter(
                nerf, medium, env, rays_o, rays_d, near, far, n_samples, detach_scatter=False
            )
            pred_img = pred.reshape(train_set.H, train_set.W, 3)
            J_img = J.reshape(train_set.H, train_set.W, 3)
            p = psnr(pred_img, rgb).item()
            l = l1(pred_img, rgb).item()
        nerf.train(); medium.train(); env.train()
        return p, l, pred_img, J_img, rgb

    pbar = tqdm(range(iters), desc="train")
    for step in pbar:
        for g in optim.param_groups:
            g["lr"] = lr_schedule(step)

        idx = torch.randint(0, len(train_set), (1,), device=device).item()
        batch = train_set[idx]
        rgb = batch["rgb"]
        rays_o = batch["rays_o"]
        rays_d = batch["rays_d"]

        r_o, r_d, target = sample_random_rays(rays_o, rays_d, rgb, batch_rays)

        pred, _, extras = volume_render_rte_single_scatter(
            nerf, medium, env, r_o, r_d, near, far, n_samples, detach_scatter=detach_scatter
        )

        loss_img = torch.mean(torch.abs(pred - target))

        z_vals = extras["z_vals"]  # (R,S)
        pts = r_o[:, None, :] + r_d[:, None, :] * z_vals[..., None]  # (R,S,3)
        T = extras["T"]  # (R,S,1)

        pts_flat = pts.reshape(-1, 3)
        Lin = env(pts_flat).reshape(pts.shape[0], pts.shape[1], 3)
        L_pred = (T * Lin).detach() if detach_scatter else (T * Lin)

        res = rte_residual_along_rays(env, medium, pts, r_d, L_pred)
        loss_rte = torch.mean(res * res)

        loss = w_img * loss_img + w_rte * loss_rte

        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

        with torch.no_grad():
            p = psnr(pred, target).item()
            sigma_t_val = torch.nn.functional.softplus(medium._sigma_t).item()
            g_val = torch.tanh(medium._g).item()

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "img": f"{loss_img.item():.4f}",
            "rte": f"{loss_rte.item():.4f}",
            "psnr": f"{p:.2f}",
            "sigma_t": f"{sigma_t_val:.3f}",
            "g": f"{g_val:.3f}",
        })

        if step % cfg["train"]["log_every"] == 0:
            writer.add_scalar("train/loss", loss.item(), step)
            writer.add_scalar("train/loss_img", loss_img.item(), step)
            writer.add_scalar("train/loss_rte", loss_rte.item(), step)
            writer.add_scalar("train/psnr", p, step)

        if step % cfg["train"]["eval_every"] == 0 and step > 0:
            ep, el, pred_img, J_img, gt = eval_one()
            writer.add_scalar("test/psnr", ep, step)
            writer.add_scalar("test/l1", el, step)
            writer.add_image("test/pred", pred_img.permute(2, 0, 1), step)
            writer.add_image("test/J", J_img.permute(2, 0, 1), step)
            writer.add_image("test/gt", gt.permute(2, 0, 1), step)

        if step % cfg["train"]["ckpt_every"] == 0 and step > 0:
            ckpt_path = os.path.join(out_dir, "ckpts", f"step_{step:06d}.pt")
            save_ckpt(
                ckpt_path,
                {"nerf": nerf.state_dict(), "medium": medium.state_dict(), "env": env.state_dict()},
                optim,
                step,
            )

    writer.close()
    print("Done. Outputs in:", out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/seathru.yaml")
    args = ap.parse_args()
    main(args.config)
