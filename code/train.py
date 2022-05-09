import argparse
import math
import random
import os
import torch

from lpips_pytorch import LPIPS, lpips

from torch import nn, autograd, optim
from torch.nn import functional as F
from torchvision import utils, models
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageColor

from dataset import get_dataloader
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix
from non_leaking import augment, AdaptiveAugment
try:
    import wandb

except ImportError:
    wandb = None

torch.cuda.empty_cache()


@torch.no_grad()
def get_image(tensor, **kwargs):
    grid = utils.make_grid(tensor, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


@torch.no_grad()
def concat_image_by_height(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None


def save_sample_images(g_ema, args, itr_idx, sample_z, sample_z_label=None):
    with torch.no_grad():
        filename = os.path.join(args.output_path, f"sample/{str(itr_idx).zfill(6)}.png")

        g_ema.eval()
        
        nof_images = sample_z.shape[0]
        minv = -1
        maxv = 1
        if sample_z_label is None or sample_z_label.shape[2] != 2:
            sample, _ = g_ema([sample_z])
            minv = torch.min(sample)
            maxv = torch.max(sample)

            utils.save_image(
                sample,
                filename,
                nrow=int(nof_images ** 0.5),
                normalize=True,
                value_range=(minv, maxv))
            return
        else:
            # For Conditional GAN, draw borders around each image according to its label
            sample, _ = g_ema([sample_z], labels=sample_z_label)
            minv = torch.min(sample)
            maxv = torch.max(sample)
            sample_z_label = sample_z_label.cpu().squeeze()
            im = get_image(
                sample,
                nrow=int(nof_images ** 0.5),
                normalize=True,
                value_range=(minv, maxv),
                padding=0)
            border_size = 4
            draw = ImageDraw.Draw(im)
            # we have to iterate, each image get a padding fill value of its own
            for i in range(nof_images):
                
                label = sample_z_label[i].numpy()[0].item()

                if label == 0:
                    row = i // 4
                    column = i % 4
                    # img = sample[i, :]
                    draw.rectangle(
                        [column*256,row*256,
                        (column+1)*256,(row+1)*256],
                        outline=ImageColor.getrgb("darkred"),
                        width=border_size)
               
            im.save(filename)


def save_real_vs_encoded(generator, args, i, sample_images, sample_labels):
    with torch.no_grad():
        latents = encoder(sample_images)
        encoded_imgs, _ = generator([latents], labels=sample_labels, input_is_latent=True)

        # save real vs. encoded images
        minv = torch.min(encoded_imgs)
        maxv = torch.max(encoded_imgs)
        encoded_image = get_image(
            encoded_imgs,
            nrow=int(args.batch ** 0.5),
            normalize=True,
            value_range=(minv, maxv)
        )

        # save real vs. encoded images
        minv = torch.min(sample_images)
        maxv = torch.max(sample_images)
        real_image = get_image(
            sample_images,
            nrow=int(args.batch ** 0.5),
            normalize=True,
            value_range=(minv, maxv)
        )
        filename_img = os.path.join(args.output_path, f"sample/real_vs_encoded_{str(i).zfill(6)}.png")
        combined = concat_image_by_height(encoded_image, real_image)
        combined.save(filename_img)


def train_encoder(args, loader, generator, g_optim, g_ema, device, classifier, encoder, e_optim, ckpt):

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    class_loss = torch.tensor(0.0, device=device)
    reconstruct_loss = torch.tensor(0.0, device=device)
    loss_dict = {}

    g_module = generator
    e_module = encoder

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, real_labels = next(loader)
        real_img, real_labels = real_img.to(device), real_labels.to(device)

        real_encoded = encoder(real_img)
        real_logits = classifier(real_img)

        _, encoded_labels = torch.max(real_logits, 1)
        encoded_labels = F.one_hot(encoded_labels, num_classes=args.classifier_nof_classes)
        fake_img, _ = generator([real_encoded], labels=encoded_labels, input_is_latent=True)
        fake_logits = classifier(fake_img)
        
        logsoft = torch.nn.LogSoftmax(dim = 1)
        class_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(logsoft(fake_logits), logsoft(real_logits))

        loss_dict["class_loss"] = class_loss

        fake_encoded = encoder(fake_img)
        reconstruct_loss_x = F.l1_loss(fake_img, real_img)
        reconstruct_loss_w = F.l1_loss(fake_encoded, real_encoded)
        reconstruct_loss_lpips = lpips(fake_img/fake_img.max(), real_img/real_img.max(), net_type='alex', version='0.1').flatten()

        reconstruct_loss = reconstruct_loss_x + reconstruct_loss_w + reconstruct_loss_lpips
        loss_dict["reconstruct_loss"] = reconstruct_loss
        loss_dict["reconstruct_loss_x"] = reconstruct_loss_x
        loss_dict["reconstruct_loss_w"] = reconstruct_loss_w
        loss_dict["reconstruct_loss_lpips"] = reconstruct_loss_lpips

        generator.zero_grad()
        encoder.zero_grad()
        total_loss = class_loss + reconstruct_loss
        loss_dict["total_loss"] = total_loss
        total_loss.backward()
        e_optim.step()
        g_optim.step()
        accumulate(g_ema, g_module, accum)
 
        class_loss_val = loss_dict["class_loss"].mean().item()
        reconstruct_loss_val = loss_dict["reconstruct_loss"].mean().item()
        reconstruct_loss_x_val = loss_dict["reconstruct_loss_x"].mean().item()
        reconstruct_loss_w_val = loss_dict["reconstruct_loss_w"].mean().item()
        reconstruct_loss_lpips_val = loss_dict["reconstruct_loss_lpips"].mean().item()
        total_loss_val = loss_dict["total_loss"].mean().item()

        if get_rank() == 0:
            print(  f"class: {class_loss_val:.6f}; "
                    f"reconstruct: {reconstruct_loss_val:.6f} "
                    f"reconstruct_x: {reconstruct_loss_x_val:.6f} "
                    f"reconstruct_w: {reconstruct_loss_w_val:.6f} "
                    f"reconstruct_lpips: {reconstruct_loss_lpips_val:.6f} "
                    f"total_loss: {total_loss_val:.6f} "
                    )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Class Loss": class_loss_val,
                        "Reconstruct Loss": reconstruct_loss_val,
                        "Reconstruct x": reconstruct_loss_x_val,
                        "Reconstruct w": reconstruct_loss_w_val,
                        "Reconstruct LPIPS": reconstruct_loss_lpips_val,
                        "Total Loss": total_loss_val
                    }
                )
                wandb.watch(generator)
                wandb.watch(encoder)

            if i == 0:
                continue
            
            if i % args.save_samples_every == 0:
                with torch.no_grad():
                    encoded_image = get_image(
                        fake_img,
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        scale_each=True
                    )
                    real_image = get_image(
                        real_img,
                        nrow=int(args.batch ** 0.5),
                        normalize=True,
                        scale_each=True
                    )
                    filename_img = os.path.join(args.output_path, f"sample/img_{str(i).zfill(6)}.png")
                    combined = concat_image_by_height(encoded_image, real_image)
                    combined.save(filename_img)

            if i % args.save_checkpoint_every == 0:
                filename = os.path.join(args.output_path, f"checkpoint/{str(i).zfill(6)}.pt")
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "e": e_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "args": args,
                    },
                    filename,
                )


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, ckpt):
    loader = sample_data(loader)

    pbar = range(args.iter)

    cgan = args.cgan

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    weighted_path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    if cgan:
        sample_z_label = torch.randint(0, args.classifier_nof_classes, (args.n_sample, 1), device=device)
        sample_z_label = F.one_hot(sample_z_label, num_classes=args.classifier_nof_classes)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        if cgan:
            real_img, real_label = next(loader)
            real_img, real_label = real_img.to(device), real_label.to(device)
            # real_label = F.one_hot(real_label, num_classes=args.classifier_nof_classes)
        else:
            real_img = next(loader)
            real_img = real_img.to(device)

        requires_grad(generator, False)
        requires_grad(discriminator, True)

        # Generate image from noise
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        if cgan:
            # Get one-hot labels from noise
            random_label = torch.randint(0, args.classifier_nof_classes, (args.batch, 1), device=device)
            random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)
            fake_img, _ = generator(noise, labels=random_label, input_is_latent=False)
        else:
            fake_img, _ = generator(noise)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img

        if cgan:
            fake_pred = discriminator(fake_img, random_label)
            real_pred = discriminator(real_img_aug, real_label)
        else:
            fake_pred = discriminator(fake_img)
            real_pred = discriminator(real_img_aug)

        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            if cgan:
                real_pred = discriminator(real_img_aug, real_label)
            else:
                real_pred = discriminator(real_img_aug)

            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

            # we're using this image again
            real_img.requires_grad = False

        loss_dict["r1"] = r1_loss

        requires_grad(generator, True)
        requires_grad(discriminator, False)

        # Generate fake image from random noise and random label
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        if cgan:
            # Get one-hot labels from noise
            random_label = torch.randint(0, args.classifier_nof_classes, (args.batch, 1), device=device)
            random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)
            fake_img, _ = generator(noise, labels=random_label, input_is_latent=False)
        else:
            fake_img, _ = generator(noise)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        if cgan:
            fake_pred = discriminator(fake_img, random_label)
        else:
            fake_pred = discriminator(fake_img)

        g_loss = g_nonsaturating_loss(fake_pred)

        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)

            if cgan:
                random_label = torch.randint(0, args.classifier_nof_classes, (path_batch_size, 1), device=device)
                random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)
                fake_img, latents = generator(noise, labels=random_label, input_is_latent=False, return_latents=True)
            else:
                fake_img, latents = generator(noise, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()

            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced["d"].mean().item()
        g_loss_val = loss_reduced["g"].mean().item()
        r1_val = loss_reduced["r1"].mean().item()
        path_loss_val = loss_reduced["path"].mean().item()
        real_score_val = loss_reduced["real_score"].mean().item()
        fake_score_val = loss_reduced["fake_score"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; "
                    f"path: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                    f"augment: {ada_aug_p:.4f}"
                )
            )

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                    }
                )

            if i == 0:
                continue

            if i % args.save_samples_every == 0:
                save_sample_images(g_ema, args, i, sample_z, sample_z_label if cgan else None)

            if i % args.save_checkpoint_every == 0:
                filename = os.path.join(args.output_path, f"checkpoint/{str(i).zfill(6)}.pt")
                torch.save(
                     {
                         "g": g_module.state_dict(),
                         "d": d_module.state_dict(),
                         "g_ema": g_ema.state_dict(),
                         "g_optim": g_optim.state_dict(),
                         "d_optim": d_optim.state_dict(),
                         "args": args,
                         "ada_aug_p": ada_aug_p
                     },
                     filename,
                )


def train_stylechexplain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, classifier, encoder, e_optim, ckpt):
    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    weighted_path_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    class_loss = torch.tensor(0.0, device=device)
    reconstruct_loss = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module
        e_module = encoder.module

    else:
        g_module = generator
        d_module = discriminator
        e_module = encoder

    accum = 0.5 ** (32 / (10 * 1000))
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0
    r_t_stat = 0

    if args.augment and args.augment_p == 0:
        ada_augment = AdaptiveAugment(args.ada_target, args.ada_length, 8, device)

    sample_z = torch.randn(args.n_sample, args.latent, device=device)
    sample_z_label = torch.randint(0, args.classifier_nof_classes, (args.n_sample, 1), device=device)
    sample_z_label = F.one_hot(sample_z_label, num_classes=args.classifier_nof_classes)

    sample_images, sample_labels = next(iter(loader))
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        real_img, real_labels = next(loader)
        real_img, real_labels = real_img.to(device), real_labels.to(device)

        #real_labels = F.one_hot(real_labels, num_classes=args.classifier_nof_classes)

        # First, train discriminator
        requires_grad(generator, False)
        requires_grad(encoder, False)
        requires_grad(discriminator, True)

        '''
        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img
        '''

        # Generate image from encoded_real
        encoded_real = encoder(real_img)
        logits = classifier(real_img)
        _, encoded_labels = torch.max(logits, 1)
        encoded_labels = F.one_hot(encoded_labels, num_classes=args.classifier_nof_classes)

        fake_img_by_encoded, _ = generator([encoded_real], labels=encoded_labels, input_is_latent=True)

        # Generate image from noise
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        random_label = torch.randint(0, args.classifier_nof_classes, (args.batch, 1), device=device)
        random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)

        fake_img_by_z, _ = generator(noise, labels=random_label, input_is_latent=False)

        # Discriminator calls
        fake_pred_by_encoded = discriminator(fake_img_by_encoded.detach(), encoded_labels.detach())
        fake_pred_by_z = discriminator(fake_img_by_z.detach(), random_label)
        real_pred = discriminator(real_img, real_labels)

        # Loss
        d_loss_1 = d_logistic_loss(real_pred, fake_pred_by_encoded)
        d_loss_2 = d_logistic_loss(real_pred, fake_pred_by_z)

        d_loss = d_loss_1 + d_loss_2

        loss_dict["d"] = d_loss
        loss_dict["real_score"] = real_pred.mean()
        loss_dict["fake_score"] = fake_pred_by_encoded.mean()
        loss_dict["fake_by_z_score"] = fake_pred_by_z.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            ada_aug_p = ada_augment.tune(real_pred)
            r_t_stat = ada_augment.r_t_stat

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.augment:
                real_img_aug, _ = augment(real_img, ada_aug_p)

            else:
                real_img_aug = real_img

            real_pred = discriminator(real_img_aug, real_labels)
            r1_loss = d_r1_loss(real_pred, real_img)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

            # we're using this image again
            real_img.requires_grad = False

        loss_dict["r1"] = r1_loss

        # Train Generator
        requires_grad(generator, True)
        requires_grad(encoder, False)
        requires_grad(discriminator, False)

        '''
        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)
        '''

        # Generate fake image from random noise and random label
        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
        random_label = torch.randint(0, args.classifier_nof_classes, (args.batch, 1), device=device)
        random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)
        fake_img_by_z, _ = generator(noise, labels=random_label, input_is_latent=False)
        fake_pred_by_z = discriminator(fake_img_by_z, random_label)

        # Generate image from encoded_real
        encoded_real = encoder(real_img)
        logits = classifier(real_img)
        _, encoded_labels = torch.max(logits, 1)
        encoded_labels = F.one_hot(encoded_labels, num_classes=args.classifier_nof_classes)
        fake_img_by_encoded, _ = generator([encoded_real.detach()], labels=encoded_labels.detach(), input_is_latent=True)
        fake_pred_by_encoded = discriminator(fake_img_by_encoded, encoded_labels)

        g_loss = (g_nonsaturating_loss(fake_pred_by_z) + g_nonsaturating_loss(fake_pred_by_encoded)) / 2
        loss_dict["g"] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        g_regularize = i % args.g_reg_every == 0

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)

            noise = mixing_noise(path_batch_size, args.latent, args.mixing, device)
            random_label = torch.randint(0, args.classifier_nof_classes, (path_batch_size, 1), device=device)
            random_label = F.one_hot(random_label, num_classes=args.classifier_nof_classes)

            fake_img_path_reg, latents = generator(noise, labels=random_label, input_is_latent=False, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img_path_reg, latents, mean_path_length
            )

            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img_path_reg[0, 0, 0, 0]

            generator.zero_grad()
            weighted_path_loss.backward()
            g_optim.step()

            mean_path_length_avg = (
                reduce_sum(mean_path_length).item() / get_world_size()
            )

        loss_dict["path"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        # Train Encoder & Generator
        requires_grad(generator, True)
        requires_grad(encoder, True)
        requires_grad(discriminator, False)

        # Generate image from real_encoded
        real_encoded = encoder(real_img)
        real_logits = classifier(real_img)
        _, encoded_labels = torch.max(real_logits, 1)
        encoded_labels = F.one_hot(encoded_labels, num_classes=args.classifier_nof_classes)
        fake_img, _ = generator([real_encoded], labels=encoded_labels, input_is_latent=True)
        fake_logits = classifier(fake_img)

        # KLDivLoss expects input to be log(prob) and target to be prob.
        logsoft = torch.nn.LogSoftmax(dim=1)
        class_loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(logsoft(fake_logits),
                                                                                logsoft(real_logits))

        loss_dict["class_loss"] = class_loss

        fake_encoded = encoder(fake_img)
        reconstruct_loss_x = F.l1_loss(fake_img, real_img)
        reconstruct_loss_w = F.l1_loss(fake_encoded, real_encoded)
        reconstruct_loss_lpips = lpips(fake_img/fake_img.max(), real_img/real_img.max(), net_type='alex', version='0.1').flatten()

        reconstruct_loss = reconstruct_loss_x + reconstruct_loss_w + reconstruct_loss_lpips
        loss_dict["reconstruct_loss"] = reconstruct_loss
        loss_dict["reconstruct_loss_x"] = reconstruct_loss_x
        loss_dict["reconstruct_loss_w"] = reconstruct_loss_w
        loss_dict["reconstruct_loss_lpips"] = reconstruct_loss_lpips

        encoder.zero_grad()
        generator.zero_grad()

        total_loss = class_loss + reconstruct_loss
        total_loss.backward()

        e_optim.step()
        g_optim.step()

        # update g_ema module
        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_dict["d"].mean().item()
        g_loss_val = loss_dict["g"].mean().item()
        r1_val = loss_dict["r1"].mean().item()
        path_loss_val = loss_dict["path"].mean().item()
        real_score_val = loss_dict["real_score"].mean().item()
        fake_score_val = loss_dict["fake_score"].mean().item()
        path_length_val = loss_dict["path_length"].mean().item()
        class_loss_val = loss_dict["class_loss"].mean().item()
        reconstruct_loss_val = loss_dict["reconstruct_loss"].mean().item()
        reconstruct_loss_x_val = loss_dict["reconstruct_loss_x"].mean().item()
        reconstruct_loss_w_val = loss_dict["reconstruct_loss_w"].mean().item()
        reconstruct_loss_lpips_val = loss_dict["reconstruct_loss_lpips"].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f"d: {d_loss_val:.2f}; g: {g_loss_val:.2f}; r1: {r1_val:.2f}; "
                    f"path: {path_loss_val:.2f}; mean path: {mean_path_length_avg:.2f}; "
                    f"augment: {ada_aug_p:.1f}; class: {class_loss_val:.2f}; "
                    f"reconstruct: {reconstruct_loss_val:.1f} "
                )
            )
            '''
            print(f"d: {d_loss_val:.2f}; g: {g_loss_val:.2f}; r1: {r1_val:.2f}; "
                    f"path: {path_loss_val:.2f}; mean path: {mean_path_length_avg:.2f}; "
                    f"augment: {ada_aug_p:.1f}; class: {class_loss_val:.2f}; "
                    f"reconstruct: {reconstruct_loss_val:.1f} "
                    f"reconstruct_x: {reconstruct_loss_x_val:.1f} "
                    f"reconstruct_w: {reconstruct_loss_w_val:.1f} "
                    f"reconstruct_lpips: {reconstruct_loss_lpips_val:.1f} "
                    )
            '''

            if wandb and args.wandb:
                wandb.log(
                    {
                        "Generator": g_loss_val,
                        "Discriminator": d_loss_val,
                        "Augment": ada_aug_p,
                        "Rt": r_t_stat,
                        "R1": r1_val,
                        "Path Length Regularization": path_loss_val,
                        "Mean Path Length": mean_path_length,
                        "Real Score": real_score_val,
                        "Fake Score": fake_score_val,
                        "Path Length": path_length_val,
                        "Class Loss": class_loss_val,
                        "Reconstruct Loss": reconstruct_loss_val,
                        "Reconstruct x": reconstruct_loss_x_val,
                        "Reconstruct w": reconstruct_loss_w_val,
                        "Reconstruct LPIPS": reconstruct_loss_lpips_val,
                    }
                )

            if i == 0:
                continue

            if i % args.save_samples_every == 0:
                save_sample_images(g_ema, args, i, sample_z, sample_z_label)
                save_real_vs_encoded(generator, args, i, sample_images, sample_labels)

            if i % args.save_checkpoint_every == 0:
                filename = os.path.join(args.output_path, f"checkpoint/{str(i).zfill(6)}.pt")
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "e": e_module.state_dict(),
                        "g_ema": g_ema.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "e_optim": e_optim.state_dict(),
                        "args": args,
                        "ada_aug_p": ada_aug_p
                    },
                    filename,
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("dataset_path", type=str, help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | autoencoder | StyleEx)')
    parser.add_argument(
        "--iter", type=int, default=800000, help="total training iterations"
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch sizes for each gpus"
    )
    parser.add_argument('--output_path',
                        type=str,
                        help='Path into which artifacts are saved (sampled images, checkpoint')

    parser.add_argument(
        "--n_sample",
        type=int,
        default=16,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--size", type=int, default=256, help="image sizes for the model"
    )
    parser.add_argument(
        "--r1", type=float, default=10, help="weight of the r1 regularization"
    )
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--wandb", action="store_true", help="use weights and biases logging"
    )
    parser.add_argument(
        "--augment", action="store_true", help="apply non leaking augmentation"
    )
    parser.add_argument(
        "--augment_p",
        type=float,
        default=0,
        help="probability of applying augmentation. 0 = use adaptive augmentation",
    )
    parser.add_argument(
        "--ada_target",
        type=float,
        default=0.6,
        help="target augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_length",
        type=int,
        default=500 * 1000,
        help="target duraing to reach augmentation probability for adaptive augmentation",
    )
    parser.add_argument(
        "--ada_every",
        type=int,
        default=256,
        help="probability update interval of the adaptive augmentation",
    )

    ####
    # Stylechexplain parameters
    ####
    parser.add_argument(
        "--classifier_nof_classes",
        type=int,
        default=2,
        help="For Stylechexplain or Conditional StyleGAN2, number of classifier target classes",
    )
    parser.add_argument(
        "--classifier_ckpt",
        type=str,
        default=None,
        help="Path for pretrained classifier checkpoint",
    )
    parser.add_argument(
        "--cgan",
        default=False,
        action="store_true",
        help="Train Conditional GAN, auto set to True for StyleEx")
    parser.add_argument(
        "--encoder_ckpt",
        type=str,
        default=None,
        help="Path for pretrained encoder checkpoint",
    )

    parser.add_argument(
        "--compare_to_healthy", action="store_true",
        help="dataset consists of one specific class and no finding/healthy samples."
    )
    parser.add_argument('--filter_label', type=str)

    args = parser.parse_args()

    # Default params
    args.latent = 512
    args.n_mlp = 8
    args.embedding_size = 10  # embedding for labels in Generator and Discriminator

    # control how often samples and checkpoints are saved
    args.save_samples_every = 500
    args.save_checkpoint_every = 500

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # make sure required output dirs exist
    dirs = ['checkpoint', 'sample']
    for d in dirs:
        pth = os.path.join(args.output_path, d)
        if not os.path.exists(pth):
            os.makedirs(pth, exist_ok=True)
    
    args.start_iter = 0

    if get_rank() == 0 and wandb is not None and args.wandb:
        wandb.init(project=args.arch, resume=True)

    loader = get_dataloader(args)

    supported_archs = ['stylegan2', 'autoencoder', 'StyleEx']
    if args.arch not in supported_archs:
        raise ValueError("Only one of %s is supported architecture" % supported_archs)

    # struct to control build of different components in architecture
    build = dict()
    if args.arch == 'stylegan2':
        from model import Generator, Discriminator
        build = {"G": True, "D": True, "E": False, "C": False}

    elif args.arch == 'autoencoder':
        from model import Generator, Encoder
        args.cgan = True
        build = {"G": True, "D": False, "E": True, "C": True}

    elif args.arch == 'StyleEx':
        from model import Generator, Discriminator, Encoder
        args.cgan = True
        build = {"G": True, "D": True, "E": True, "C": True}
    else:
        print("Architecture %s not supported, exiting.." % args.arch)
        exit()

    ckpt = None
    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

    # Build Generator
    if build["G"]:
        generator = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan, nof_classes=args.classifier_nof_classes, embedding_size=args.embedding_size
        ).to(device)

        g_ema = Generator(
            args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier,
            conditional_gan=args.cgan, nof_classes=args.classifier_nof_classes, embedding_size=args.embedding_size
        ).to(device)
        g_ema.eval()
        accumulate(g_ema, generator, 0)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )

        if ckpt:
            generator.load_state_dict(ckpt["g"])
            g_ema.load_state_dict(ckpt["g_ema"])
            g_optim.load_state_dict(ckpt["g_optim"])

        if args.distributed:
            generator = nn.parallel.DistributedDataParallel(
                generator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

    # Build Discriminator
    if build["D"]:
        discriminator = Discriminator(
            args.size, channel_multiplier=args.channel_multiplier, conditional_gan=args.cgan
        ).to(device)

        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )

        if ckpt:
            discriminator.load_state_dict(ckpt["d"])
            d_optim.load_state_dict(ckpt["d_optim"])

        if args.distributed:
            discriminator = nn.parallel.DistributedDataParallel(
                discriminator,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

    # Build Classifier
    if build["C"]:
        if args.classifier_ckpt is None:
            print("Pretrianed classifier needed!")
            exit()

        # classifier = init_ffhq_classifier(device, args.classifier_ckpt)
        classifier = models.densenet121(pretrained=True)
        classifier.classifier = torch.nn.Linear(1024, args.classifier_nof_classes)
        classifier.to(device)

        print("loading classifier:", args.classifier_ckpt)
        classifier.load_state_dict(torch.load(args.classifier_ckpt))

        # Classifier shouldn't be trained
        classifier.eval()
        requires_grad(classifier, False)

    # Build Encoder
    if build["E"]:
        encoder = Encoder(
            args.size, channel_multiplier=args.channel_multiplier, output_channels=args.latent
        ).to(device)
        e_optim = optim.Adam(
            encoder.parameters(),
            lr=args.lr
        )

        if args.encoder_ckpt is not None:
            # Encoder ckpt given separately
            print("load encoder model:", args.encoder_ckpt)
            e_ckpt = torch.load(args.encoder_ckpt, map_location=lambda storage, loc: storage)
        else:
            # Encoder given as part of general ckpt
            e_ckpt = ckpt

        if e_ckpt is not None and "e" in e_ckpt:
            encoder.load_state_dict(e_ckpt["e"])
            e_optim.load_state_dict(e_ckpt["e_optim"])

        if args.distributed:
            encoder = nn.parallel.DistributedDataParallel(
                encoder,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                broadcast_buffers=False,
            )

    if args.arch == "stylegan2":
        train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device, ckpt)

    elif args.arch == "autoencoder":
        train_encoder(args, loader, generator, g_optim, g_ema, device, classifier, encoder, e_optim, ckpt)

    elif args.arch == "StyleEx":
        train_stylechexplain(args, loader, generator, discriminator, g_optim, d_optim, g_ema, device,
                             classifier, encoder, e_optim, ckpt)
