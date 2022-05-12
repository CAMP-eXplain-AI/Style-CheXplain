import torch
from torchvision import transforms, models, utils
from model import Generator, Encoder
from tqdm import tqdm
import os
import collections
from timer import timer
import datetime

from dataset import get_dataloader
import att_find_code.att_find_functions as att_find_func


class Args:
    pass


if __name__ == "__main__":
    args = Args()
    args.device = "cuda"
    args.batch_size = 16

    # input
    args.ckpt = "styleex_training/cardio_vs_all/checkpoint/062000.pt"
    args.classifier_ckpt = "cardiomegaly_model.pth"

    # output
    features_save_file_path = "f_cardio_vs_all1.pt"
    change_decision_coords_save_file_path = "c_cardio_vs_all.pt"
    args.output_path = "att_find_artifacts"

    # dataset params
    args.dataset_path = "./mdb-gender/val/"
    args.filter_label = "Cardiomegaly"
    args.compare_to_healthy = False

    # search params
    args.source = "noise"  # should be one of ['noise', 'data']
    args.num_of_batches = 16  # how many search iterations
    s_shift_size = 2.5

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    torch.set_grad_enabled(False)
    torch.manual_seed(42)

    # load generator
    ckpt = torch.load(args.ckpt)
    params = ckpt["args"]

    generator = Generator(params.size, params.latent, params.n_mlp,
                          channel_multiplier=params.channel_multiplier,
                          conditional_gan=params.cgan if 'cgan' in params else False,
                          nof_classes=params.classifier_nof_classes if 'classifier_nof_classes' in params else False,
                          embedding_size=params.embedding_size
                          ).to(args.device)

    generator.load_state_dict(ckpt["g_ema"], strict=False)
    generator.eval()

    # load classifier
    classifier = models.densenet121(pretrained=True)
    classifier.classifier = torch.nn.Linear(1024, 2)
    classifier.load_state_dict(torch.load(args.classifier_ckpt))
    classifier.to(args.device)
    classifier.eval()

    # get latents for noise or images
    dlatents_, labels_ = [], []
    if args.source == "data":
        # load encoder
        encoder = Encoder(params.size, channel_multiplier=params.channel_multiplier,
                          output_channels=params.latent
                          ).to(args.device)
        encoder.load_state_dict(ckpt["e"], strict=False)
        encoder.eval()

        dataloader = get_dataloader(args)

        for i, (images, labels) in enumerate(dataloader):
            if i == args.num_of_batches:
                break
            dlatents_.append(encoder(images.to(args.device)))
            labels_.append(labels)

        labels = torch.cat(labels_)

    elif args.source == "noise":
        for i in range(args.num_of_batches):
            noise = torch.rand((args.batch_size, params.latent)).to(args.device)
            latent = generator.get_latent(noise)
            dlatents_.append(latent)

    dlatents = torch.cat(dlatents_)

    affines = att_find_func.get_affines(generator)

    # get style vectors per latent
    values_per_index = collections.defaultdict(list)
    for dlatent in dlatents:
        s_img = torch.cat(att_find_func.get_style_for_dlantent(dlatent, affines)).cpu().numpy()

        for i, s_val in enumerate(s_img):
            values_per_index[i].append(s_val)
    values_per_index = dict(values_per_index)

    # min and max value per coordinate
    s_indices_num = len(values_per_index.keys())
    minimums = [min(values_per_index[i]) for i in range(s_indices_num)] 
    maximums = [max(values_per_index[i]) for i in range(s_indices_num)]

    LAYER_SHAPES = att_find_func.get_layer_shapes(affines)

    def process_dlatent(features, change_decision_coords, generator, affines, dlatent, label, dlatent_index,
                        show_progress=True):
        # get logit for original image
        base_prob = att_find_func.get_classifier_results(generator, classifier, dlatent, label, args.device)
        classifier_results = []
        print("Processing", dlatent_index)

        if show_progress:
            iter = tqdm(range(0, s_indices_num))
        else:
            iter = range(0, s_indices_num)

        # iterate over coordinates
        for sindex in iter:
            layer_idx, weight_idx = att_find_func.sindex_to_layer_idx_and_index(LAYER_SHAPES, sindex)
            layer = affines[layer_idx]

            # how much should we move this coordinate
            s_vals = torch.cat(att_find_func.get_style_for_dlantent(dlatent, affines))
            s_shift_down = (minimums[sindex] - s_vals[sindex]) * s_shift_size
            s_shift_up = (maximums[sindex] - s_vals[sindex]) * s_shift_size

            with torch.no_grad():
                # try shift down
                layer.bias[weight_idx] += s_shift_down
                res = att_find_func.get_classifier_results(generator, classifier, dlatent, label, args.device)
                if torch.argmax(base_prob) != torch.argmax(res):
                    if sindex not in change_decision_coords:
                        change_decision_coords[sindex] = []
                    change_decision_coords[sindex].append((dlatent_index, s_shift_down))
                    print("Decision boundary change: image", dlatent_index, "coord ", sindex, "by ", s_shift_down)

                classifier_results.extend(res - base_prob)
                layer.bias[weight_idx] -= s_shift_down

                # try shift up
                layer.bias[weight_idx] += s_shift_up
                res = att_find_func.get_classifier_results(generator, classifier, dlatent, label, args.device)

                if torch.argmax(base_prob) != torch.argmax(res):
                    if sindex not in change_decision_coords:
                        change_decision_coords[sindex] = []
                    change_decision_coords[sindex].append((dlatent_index, s_shift_up))
                    print("Decision boundary change: image", dlatent_index, "coord ", sindex, "by ", s_shift_up)

                classifier_results.extend(res - base_prob)
                layer.bias[weight_idx] -= s_shift_up

        feature = dict()
        feature['base_prob'] = base_prob
        feature['dlatent'] = dlatent
        feature['result'] = classifier_results
        feature['label'] = label
        features[dlatent_index] = feature
        if not show_progress:
            print("Dlatent", dlatent_index, "processed.")

        return features


    # calculate coordinate changes for all latents
    features = dict()
    change_decision_coords = {}

    item_list = zip(dlatents, labels) if args.source == "data" else dlatents
    for dlatent_index, item in enumerate(item_list):
        # make this work for both when we have labels and not
        if args.source == "data":
            (dlatent, label) = item
        else:
            dlatent = item
            label = None

        with timer() as t:
            process_dlatent(features, change_decision_coords, generator, affines, dlatent, label, dlatent_index, False)
            sec = int(t.elapse)
            print(f"Processed in {str(datetime.timedelta(seconds=sec))}")

        if dlatent_index % 16 == 15:
            torch.save(features, features_save_file_path)
            torch.save(change_decision_coords, change_decision_coords_save_file_path)

    torch.save(features, features_save_file_path)
    torch.save(change_decision_coords, change_decision_coords_save_file_path)
