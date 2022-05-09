import torch
from dataset import get_dataset
from torchvision import transforms, models, utils
from model import Generator, Encoder, Discriminator
import argparse
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
import os
import numpy as np
from torch.nn import functional as F
import math


@torch.no_grad()
def get_logits_latents(dataset, classifier, encoder, generator, args):
    """
    return two dictionaries in which the keys are the indices of images in dataset
    logits: values are the classifier logits output for images
    latents: values are the encoder ouputs for images
    """

    logits, latents = dict(), dict()
    indices = [idx for idx in range(args.n_sample)]

    if args.source == "data":
        # generate from data
        for idx in tqdm(indices):
            img, _ = dataset[idx]

            img = img.to(args.device)
            img = img.unsqueeze(0)

            out = classifier(img)
            logits[idx] = out

            # get latent for image using the encoder
            w = encoder(img)
            latents[idx] = w
    else:
        # generate from noise
        for idx in tqdm(indices):
            sample_z = torch.randn(1, 512, device=args.device)

            img, _ = generator([sample_z], input_is_latent=False)
            out = classifier(img)
            logits[idx] = out

            # get latent for image using generator mapping network
            w = generator.get_latent(sample_z)
            latents[idx] = w

    return logits, latents


@torch.no_grad()
def get_logits_diff(generator, classifier, eigvec, filterd_idx, logits, latents, args):
    logits_diff = dict()

    print("Checking %d styles..." % args.nof_eigenv)

    for eigenvec_idx in range(args.nof_eigenv):
        print(" Style %d" % eigenvec_idx)
        direction = args.degree * eigvec[:, eigenvec_idx]

        for idx in tqdm(filterd_idx):
            changed_latent = torch.stack([latents[idx] + direction, latents[idx] - direction]).squeeze(1)
            changed_latent.to(args.device)

            changed_img, _ = generator([changed_latent], input_is_latent=True)
            changed_logit = classifier(changed_img)

            logit_diff_pos = changed_logit[0] - logits[idx].squeeze(0)
            logit_diff_neg = changed_logit[1] - logits[idx].squeeze(0)

            logits_diff[(idx, eigenvec_idx)] = [logit_diff_pos, logit_diff_neg]

    return logits_diff


def get_logits_means(nof_eigenv, logits_diff, class_label, nof_images):
    # calc means
    means = {(eigen_idx, 'pos'): 0 for eigen_idx in range(nof_eigenv)}
    means.update({(eigen_idx, 'neg'): 0 for eigen_idx in range(nof_eigenv)})

    for (_, eigen_idx), diff in logits_diff.items():
        diff_pos = diff[0]
        means[(eigen_idx, 'pos')] += diff_pos[class_label].item()

        diff_neg = diff[1]
        means[(eigen_idx, 'neg')] += diff_neg[class_label].item()

    for key in means.keys():
        means[key] /= nof_images

    # Get rid of inconsistent changes
    for eigen_idx in range(nof_eigenv):
        if means[(eigen_idx, 'pos')] > 0 and means[(eigen_idx, 'neg')] > 0:
            means[(eigen_idx, 'pos')] = 0
            means[(eigen_idx, 'neg')] = 0
    return means


def get_most_significant_eigens_and_direct(logits_means, args):
    sort_significant_eigendirect = sorted(logits_means.items(), key=lambda x: x[1], reverse=True)[
                                   :args.nof_significant_eigenv]
    sort_significant_eigendirect = dict(sort_significant_eigendirect)

    print("EV\tDirection\tDiff")
    for (eigen, sign) in sort_significant_eigendirect:
        print("%d\t%s\t%.2f" % (eigen, sign, sort_significant_eigendirect[(eigen, sign)]))

    return sort_significant_eigendirect


def att_find(eigvec, generator, encoder, classifier, dataset, args):
    """

    :param eigvec:
    :param generator:
    :param encoder:
    :param classifier:
    :param dataset:
    :param args:
    :return:
    """

    logits, latents = get_logits_latents(dataset, classifier, encoder, generator, args)

    # keep only images with label != args.class_label.
    # FIXME- works only for two classes 
    other_label = 1 - args.class_label

    # create list of image indices which are not classified as 'class_label'
    filterd_idx = set()
    for i, l in logits.items():
        l = l.detach().cpu().squeeze().numpy().tolist()
        if l[args.class_label] < l[other_label]:
            filterd_idx.add(i)

    max_num_images = len(filterd_idx)
    print("%d images with label %d will be processed" % (max_num_images, other_label))

    # calculate diff between original and changed image logits for each eigenvector
    logits_diff = get_logits_diff(generator, classifier, eigvec, filterd_idx, logits, latents, args)

    # get mean logit diffs per eigenvector
    logits_means = get_logits_means(args.nof_eigenv, logits_diff, args.class_label, len(filterd_idx))

    # get top args.nof_significant_eigenv eigenvectors
    most_significant_eigendirect = get_most_significant_eigens_and_direct(logits_means, args)

    result_explained_images = dict()
    filterd_image_idx = filterd_idx.copy()

    for (eigen_idx, eigen_direct), diff in most_significant_eigendirect.items():
        explained_image_idx = dict()
        sign_idx = 0 if (eigen_direct == 'pos') else 1
        for image_idx in filterd_image_idx:
            logits_change_diff = logits_diff[(image_idx, eigen_idx)]
            logits_change = logits_change_diff[sign_idx][args.class_label]

            if logits_change > args.change_threshold:
                explained_image_idx[image_idx] = logits_change

        if len(explained_image_idx) == max_num_images:
            print('Eigenvector %d:%s explained all images' % (eigen_idx, eigen_direct))
            result_explained_images[(eigen_idx, eigen_direct)] = explained_image_idx
            filterd_image_idx = filterd_image_idx - explained_image_idx.keys()
            break

        if len(explained_image_idx) == 0:
            print('Eigenvector %d:%s explained no images' % (eigen_idx, eigen_direct))
            continue

        result_explained_images[(eigen_idx, eigen_direct)] = explained_image_idx
        print('Eigenvector %d:%s explained %d images' % (eigen_idx, eigen_direct, len(explained_image_idx)))

        filterd_image_idx = filterd_image_idx - explained_image_idx.keys()

        if len(filterd_image_idx) == 0:
            print('100% of images explained')
            break

    unexplained_count = len(filterd_image_idx)
    if unexplained_count != 0:
        explained_count = max_num_images - unexplained_count
        explained_frac = explained_count / max_num_images
        print("%.2f%% (%d/%d) of images explained" % (explained_frac, explained_count, max_num_images))

    return result_explained_images, logits, latents


def save_explained_images(result_explained_images, dataset, num_images):
    for (eigen_idx, eigen_direct), exp_image_idx in result_explained_images.items():
        filename = str(eigen_idx) + '_' + str(eigen_direct) + '.png'
        sampled_explained_images = []
        i = 0
        for explained_idx in exp_image_idx.keys():
            sampled_explained_images.append(dataset[explained_idx][0])
            i += 1
            if i == num_images:
                break
        utils.save_image(
            sampled_explained_images,
            filename,
            nrow=int(num_images ** 0.5),
            normalize=True,
            value_range=(-1, 1),
        )


def write_prob(img_tensor, prob):
    # write probability on each image
    font_file = "ARIALUNI.ttf"
    fill_color = (0, 0, 255)  # blue
    txt_font = ImageFont.truetype(font_file, 20)

    for i in range(img_tensor.shape[0]):
        # this is a hack
        image = utils.make_grid(img_tensor[i, :], int=1, normalize=True)
        im_arr = image.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        img_to_draw = Image.fromarray(im_arr)
        draw = ImageDraw.Draw(img_to_draw)
        draw.multiline_text((8, 8), ('%.2f' % (prob[i])), font=txt_font, fill=fill_color)

        out_image = torch.from_numpy(np.array(img_to_draw)).permute(2, 0, 1).to(dtype=torch.uint8)

        img_tensor[i, :] = out_image

    return img_tensor


def get_probablities(set_logits, set_logits_diffs, class_label):
    # convert logits to porbs
    logits_before = [l.squeeze(0) for l in set_logits]
    logits_before = [l[class_label] for l in logits_before]
    prob_before = [torch.sigmoid(l) for l in logits_before]

    # sum logits to get prob. after change
    logits_sum = [l1 + l2 for l1, l2 in zip(logits_before, set_logits_diffs)]
    prob_after = [torch.sigmoid(l) for l in logits_sum]

    return prob_before, prob_after


def get_discriminator_scores(d, img_before, img_after, batch_size, args):
    before_labels = torch.tensor([args.class_label], device=args.device)
    before_labels = before_labels.repeat([batch_size, 1])
    before_labels = F.one_hot(before_labels, num_classes=2)
    d_res_before = F.softplus(d(img_before, labels=before_labels)).mean().item()

    other_label = 1 - args.class_label
    after_labels = torch.tensor([other_label], device=args.device)
    after_labels = after_labels.repeat([batch_size, 1])
    after_labels = F.one_hot(after_labels, num_classes=2)
    d_res_after = F.softplus(d(img_after, labels=after_labels)).mean().item()

    return d_res_before, d_res_after


# Function to check
# Log base 2
def log2(x):
    return (math.log10(x) /
            math.log10(2))


# Function to check
# if x is power of 2
def isPowerOfTwo(n):
    return math.ceil(log2(n)) == math.floor(log2(n))


@torch.no_grad()
def save_changed_images(result_explained_images, generator, discriminator, eigvec, logits, latents,
                        num_images, dataset, args):
    g = generator
    d = discriminator
    for (eigen_idx, eigen_direct), exp_image_idx in result_explained_images.items():
        num_images = len(exp_image_idx) if len(exp_image_idx) < num_images else num_images

        # sort images by their diff, the most "explained" image first
        image_idx_set = sorted(exp_image_idx, key=exp_image_idx.get, reverse=True)[:num_images]

        # get latents for image set
        latent = [latents.get(key) for key in image_idx_set]
        latent = torch.stack(latent).squeeze(1)
        latent.to(args.device)

        direction = args.degree * eigvec[:, eigen_idx].unsqueeze(0)
        change_direct = 1 if eigen_direct == 'pos' else -1

        img, _ = g(
            [latent],
            input_is_latent=True
        )
        img1, _ = g(
            [latent + direction * change_direct],
            input_is_latent=True,
        )

        '''
        if isPowerOfTwo(latent.shape[0]):
            d_res_before, d_res_after = get_discriminator_scores(d, img, img1, latent.shape[0], args)
            print("Eigen %i D score before %.2f, D score after %.2f" % (eigen_idx, d_res_before, d_res_after))
        '''
        
        # get class prob. before and after change
        set_logits = [logits.get(key) for key in image_idx_set]
        set_logits_diffs = [exp_image_idx.get(key) for key in image_idx_set]
        prob_before, prob_after = get_probablities(set_logits, set_logits_diffs, args.class_label)

        # write after change probabilities on the images
        img1 = write_prob(img1, prob_after)

        # save also original images, when generating from dataset
        if args.source == "data":
            # get originals for image set
            origs = [dataset[idx][0] for idx in image_idx_set]
            origs = torch.stack(origs).to(args.device)
            origs = write_prob(origs, prob_before)
            img = torch.cat([origs, img, img1], 0)
        else:
            img = write_prob(img, prob_before)
            img = torch.cat([img, img1], 0)

        filename = os.path.join(args.output_path, args.cache_name + f"_index-{str(eigen_idx)}_degree-{args.degree}.png")
        utils.save_image(
            img,
            filename,
            normalize=True,
            scale_each=True,
            nrow=num_images,
        )

@torch.no_grad()
def save_changed_images_for_survey(result_explained_images, generator, discriminator, eigvec, logits, latents,
                        num_images, dataset, args):
    g = generator
    d = discriminator
    for (eigen_idx, eigen_direct), exp_image_idx in result_explained_images.items():
        num_images = len(exp_image_idx) if len(exp_image_idx) < num_images else num_images

        # sort images by their diff, the most "explained" image first
        image_idx_set = sorted(exp_image_idx, key=exp_image_idx.get, reverse=True)[:num_images]

        # get latents for image set
        latent = [latents.get(key) for key in image_idx_set]
        latent = torch.stack(latent).squeeze(1)
        latent.to(args.device)

        direction = args.degree * eigvec[:, eigen_idx].unsqueeze(0)
        change_direct = 1 if eigen_direct == 'pos' else -1

        img, _ = g(
            [latent],
            input_is_latent=True
        )
        img1, _ = g(
            [latent + direction * change_direct],
            input_is_latent=True,
        )

        '''
        if isPowerOfTwo(latent.shape[0]):
            d_res_before, d_res_after = get_discriminator_scores(d, img, img1, latent.shape[0], args)
            print("Eigen %i D score before %.2f, D score after %.2f" % (eigen_idx, d_res_before, d_res_after))
        '''
        
        # get class prob. before and after change
        set_logits = [logits.get(key) for key in image_idx_set]
        set_logits_diffs = [exp_image_idx.get(key) for key in image_idx_set]
        prob_before, prob_after = get_probablities(set_logits, set_logits_diffs, args.class_label)

        # save also original images, when generating from dataset
        if args.source == "data":
            # get originals for image set
            origs = [dataset[idx][0] for idx in image_idx_set]
            origs = torch.stack(origs).to(args.device)
            img_cat = torch.cat([origs, img, img1], 0)
        else:
            img_cat = torch.cat([img, img1], 0)

        filename = os.path.join(args.output_path, args.cache_name + f"_index-{str(eigen_idx)}_degree-{args.degree}.png")
        utils.save_image(
            img_cat,
            filename,
            normalize=True,
            scale_each=True,
            nrow=num_images,
        )

        EV_change_folder = os.path.join(args.output_path, args.cache_name + f"_image_original_vs_changed")
        if not os.path.exists(EV_change_folder):
            os.makedirs(EV_change_folder)
        original_image_file = os.path.join(EV_change_folder, args.cache_name + f"_original_index-{str(eigen_idx)}_degree-{args.degree}.png")
        changed_image_file = os.path.join(EV_change_folder, args.cache_name + f"_changed_index-{str(eigen_idx)}_degree-{args.degree}.png")
        utils.save_image(
            img,
            original_image_file,
            normalize=True,
            scale_each=True,
            nrow=num_images,
        )
        utils.save_image(
            img1,
            changed_image_file,
            normalize=True,
            scale_each=True,
            nrow=num_images,
        )
        


        individual_images_folder = os.path.join(args.output_path, args.cache_name + f"_images_after_change_EV_index-{str(eigen_idx)}_degree-{args.degree}")
        if not os.path.exists(individual_images_folder):
            os.makedirs(individual_images_folder)

        for num, changed_img in enumerate(img1):
            individual_img_file_name = os.path.join(individual_images_folder, args.cache_name + '_' + str(num) + f"_EV_index-{str(eigen_idx)}_degree-{args.degree}.png")
            utils.save_image(
                    changed_img,
                    individual_img_file_name,
                    normalize=True
                )
        for num, original_img in enumerate(img):
            original_individual_img_file_name = os.path.join(individual_images_folder, 'original_' + args.cache_name + '_' + str(num) + f"_EV_index-{str(eigen_idx)}_degree-{args.degree}.png")
            utils.save_image(
                    original_img,
                    original_individual_img_file_name,
                    normalize=True
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract Classifier Specific Attributes")
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Generator & Encoder checkpoints, we got from train.py")
    parser.add_argument(
        "--classifier_ckpt", type=str, required=True, help="Classifier checkpoint")
    parser.add_argument(
        "--source", type=str, default="data", help="Source for images either data or noise",
    )
    parser.add_argument(
        "--class_label", type=int, default=1, help="which class label to work on"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )

    # eigenvecotrs
    parser.add_argument(
        "--factor", type=str, required=True, help="Path for closed form factorization result factor file",
    )
    parser.add_argument(
        "--degree", type=int, default=5, help="Degree along eigenvector to move"
    )
    parser.add_argument(
        "--nof_eigenv", type=float, default=20, help="Number of eigenvectors to check"
    )
    parser.add_argument(
        "--nof_significant_eigenv", type=float, default=8, help="Number of most significant eigenvectors to check"
    )

    # generate from dataset
    parser.add_argument(
        "--dataset_path", type=str, default=None, help="Path for dataset MDB file",
    )
    parser.add_argument(
        "--batch", type=int, default=16, help="batch size"
    )

    # generate from noise
    parser.add_argument(
        "--n_sample", type=int, default=200, help="Number of images to generate, mandatory for noise generation"
    )

    # save results
    parser.add_argument(
        "--cache_name", type=str, default=None, help="Prefix to use for cache file names",
    )

    # search param
    parser.add_argument(
        "--change_threshold", type=float, default=10, help="Threshold of logits change for finding explained images"
    )

    parser.add_argument(
        "--save_for_survey",
        default=False,
        action="store_true",
        help="Save individual changed images for survey or other usages")

    args = parser.parse_args()

    args.output_path = "att_find_artifacts"
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    torch.set_grad_enabled(False)

    # load eigenvectors
    eigvec = torch.load(args.factor)["eigvec"].to(args.device)

    if args.nof_eigenv > eigvec.shape[1]:
        print("nof_eigenv provided is larger than available eigenvectors %d" % eigvec.shape[1])
        exit()

    # load generator & encoder
    ckpt = torch.load(args.ckpt)
    params = ckpt["args"]

    generator = Generator(params.size, params.latent, params.n_mlp, channel_multiplier=params.channel_multiplier,
                          conditional_gan=params.cgan if 'cgan' in params else False,
                          nof_classes=params.classifier_nof_classes if 'classifier_nof_classes' in params else False,
                          embedding_size=params.embedding_size).to(args.device)

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    discriminator = Discriminator(params.size, channel_multiplier=params.channel_multiplier,
                                  conditional_gan=params.cgan if 'cgan' in params else False).to(args.device)
    discriminator.load_state_dict(ckpt["d"])
    discriminator.eval()

    # load classifier
    classifier = models.densenet121(pretrained=True)
    classifier.classifier = torch.nn.Linear(1024, 2)
    classifier.load_state_dict(torch.load(args.classifier_ckpt))
    classifier.to(args.device)
    classifier.eval()

    if args.source == "data":
        encoder = Encoder(params.size, channel_multiplier=params.channel_multiplier, output_channels=params.latent
                          ).to(args.device)
        encoder.load_state_dict(ckpt["e"], strict=False)
        encoder.eval()

        if args.dataset_path is None:
            print("Dataset path is required in data mode")
            exit()

        # load dataset
        dataset = get_dataset(args)
        args.n_sample = len(dataset)
    else:
        dataset = None
        encoder = None

    result_explained_images, logits, latents = att_find(eigvec, generator, encoder, classifier, dataset, args)

    # num_saved_sample_images = 16
    # save_explained_images(result_explained_images, dataset, num_saved_sample_images)

    num_saved_changed_images = 8
    if args.save_for_survey == True:
        save_changed_images_for_survey(result_explained_images, generator, discriminator, eigvec, logits, latents,
                        num_saved_changed_images, dataset, args)
    else:
        save_changed_images(result_explained_images, generator, discriminator, eigvec, logits, latents,
                        num_saved_changed_images, dataset, args)
