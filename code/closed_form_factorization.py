import argparse

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract factor/eigenvectors of latent spaces using closed form factorization"
    )

    parser.add_argument(
        "--out", type=str, default="factor.pt", help="name of the result factor file"
    )
    parser.add_argument("ckpt", type=str, help="name of the model checkpoint")

    args = parser.parse_args()

    ckpt = torch.load(args.ckpt)

    modulate = dict()
    for k, v in ckpt["g_ema"].items():
        if all(x in k for x in ["modulation", "weight"]) and "to_rgb" not in k:
            modulate[k] = v
    
    weight_mat = []
    for k, v in modulate.items():
        weight_mat.append(v)

    W = torch.cat(weight_mat, 0)
    _, _, V = torch.svd(W + 1e-4*W.mean())

    eigvec = V.to("cpu")

    torch.save({"ckpt": args.ckpt, "eigvec": eigvec}, args.out)

