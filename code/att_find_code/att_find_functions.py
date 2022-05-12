import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from io import BytesIO
from torchvision import utils
import IPython.display


def get_affines(generator):
    affines = list()
    affines.append(generator.conv1.conv.modulation)
    for rgb in generator.convs:
        affines.append(rgb.conv.modulation)
    return affines


def get_layer_shapes(affines):
    layer_shapes = []
    for aff in affines:
        layer_shapes.append(aff.bias.shape[0])
    return layer_shapes


def get_style_for_dlantent(dlatent, affines):
    res = []
    for aff in affines:
        res.append(aff(dlatent))
    return res


def sindex_to_layer_idx_and_index(layer_shapes, sindex):
    layer_shapes_cumsum = np.concatenate([[0], np.cumsum(layer_shapes)])
    layer_idx = (layer_shapes_cumsum <= sindex).nonzero()[0][-1]
    return layer_idx, sindex - layer_shapes_cumsum[layer_idx]


def generate_image_dlatent(generator, latent, label, device):
    i = torch.unsqueeze(latent, 0).to(device)
    if label is not None:
        l = torch.unsqueeze(label, 0).to(device)
    else:
        l = None
    image = generator([i], labels=l, input_is_latent=True, randomize_noise=False)[0]
    return image


def get_classifier_results(generator, classifier, latent, label, device):
    image = generate_image_dlatent(generator, latent, label, device)
    return classifier(image)[0]


def get_image(image, normalize=True, scale_each=True, **kwargs):
    grid = utils.make_grid(image, normalize=normalize, scale_each=scale_each, **kwargs)
    # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    return im


def show_image(image, format='png', normalize=True, scale_each=True, **kwargs):
    im = get_image(image, normalize=normalize, scale_each=scale_each, **kwargs)
    bytes_io = BytesIO()
    im.save(bytes_io, format)
    IPython.display.display(IPython.display.Image(data=bytes_io.getvalue()))


def show_image_from_gen(generator, latent, label, device):
    i = torch.unsqueeze(latent, 0).to(device)
    if label is not None:
        l = torch.unsqueeze(label, 0).to(device)
    else:
        l = None
    show_image(generator([i], input_is_latent=True, labels=l, randomize_noise=False)[0][0])


### START Functions from paper ###
def filter_unstable_images(style_change_effect: np.ndarray,
                           effect_threshold: float = 0.3,
                           num_indices_threshold: int = 750) -> np.ndarray:
    """Filters out images which are affected by too many S values."""
    print(np.sum(np.abs(style_change_effect) > effect_threshold, axis=(1, 2, 3)))
    unstable_images = (
            np.sum(np.abs(style_change_effect) > effect_threshold, axis=(1, 2, 3)) >
            num_indices_threshold)
    style_change_effect[unstable_images] = 0
    return style_change_effect


def find_significant_styles(
        style_change_effect: np.ndarray,
        num_indices: int,
        class_index: int,
        max_image_effect: float = 0.2,
        sindex_offset: int = 0):
    """Returns indices in the style vector which affect the classifier.

    Args:
        style_change_effect: A shape of [num_images, 2, style_size, num_classes].
        The effect of each change of style on specific direction on each image.
        num_indices: Number of styles in the result.
        class_index: The index of the class to visualize.
        max_image_effect: Ignore contributions of styles if the previously found
        styles changed the probability of the image by more than this threshold.
        change allowed in the discriminator prediction.
        sindex_offset: The offset of the style index if style_change_effect contains
        some of the layers and not all styles.
    """

    num_images = style_change_effect.shape[0]
    style_effect_direction = np.maximum(
        0, style_change_effect[:, :, :, class_index].reshape((num_images, -1)))

    images_effect = np.zeros(num_images)
    all_sindices = []
    while len(all_sindices) < num_indices:
        next_s = np.argmax(
            np.mean(
                style_effect_direction[images_effect < max_image_effect], axis=0))

        all_sindices.append(next_s)
        images_effect += style_effect_direction[:, next_s]
        style_effect_direction[:, next_s] = 0

    return [(x // style_change_effect.shape[2],
             (x % style_change_effect.shape[2]) + sindex_offset)
            for x in all_sindices]


def draw_on_image(image, number: float,
                  font_file: str,
                  font_fill=(255, 0, 0)) -> np.ndarray:
    """Draws a number in the top left corner of the image."""
    fnt = ImageFont.truetype(font_file, 20)
    out_image = get_image(image)
    draw = ImageDraw.Draw(out_image)
    draw.multiline_text((10, 10), ('%.3f' % number), font=fnt, fill=font_fill)
    return torch.tensor(np.array(out_image))


def generate_change_image_given_dlatent(
        dlatent,
        label,
        additional_data_tuple,
        generator,
        classifier,
        class_index: int,
        sindex: int,
        s_style_min: float,
        s_style_max: float,
        style_direction_index: int,
        shift_size: float,
        label_size: int = 2,
):
    """Modifies an image given the dlatent on a specific S-index.

    Args:
        dlatent: The image dlatent, with sape [dlatent_size].
        generator: The generator model. Either StyleGAN or GLO.
        classifier: The classifier to visualize.
        class_index: The index of the class to visualize.
        sindex: The specific style index to visualize.
        s_style_min: The minimal value of the style index.
        s_style_max: The maximal value of the style index.
        style_direction_index: If 0 move s to it's min value otherwise to it's max
        value.
        shift_size: Factor of the shift of the style vector.
        label_size: The size of the label.

    Returns:
        The image after the style index modification, and the output of
        the classifier on this image.
    """
    # network_inputs = generator.style_vector_calculator(dlatent)
    affines, layer_shapes, device = additional_data_tuple
    style_vector = torch.concat(get_style_for_dlantent(dlatent, affines))
    # print(style_vector.shape, sindex)
    orig_value = style_vector[sindex]
    target_value = (s_style_min if style_direction_index == 0 else s_style_max)

    weight_shift = shift_size * (target_value - orig_value)

    layer_idx, in_idx = sindex_to_layer_idx_and_index(layer_shapes, sindex)
    with torch.no_grad():
        layer = affines[layer_idx]
        layer.bias[in_idx] += weight_shift

        # layer_one_hot = tf.expand_dims(
        #     tf.one_hot(in_idx, network_inputs[0][layer_idx].shape[1]), 0)
        # layer = generator.style_vector_calculator.style_dense_blocks[layer_idx]
        # layer.dense_bias.weights[0].assign_add(weight_shift * layer_one_hot)
        # network_inputs = generator.style_vector_calculator(dlatent)
        images_out = generate_image_dlatent(generator, dlatent, label, device)
        layer.bias[in_idx] -= weight_shift

        # layer.dense_bias.weights[0].assign_add(-weight_shift * layer_one_hot)

        # images_out = tf.maximum(tf.minimum(images_out, 1), -1)
        # change_image = tf.transpose(images_out, [0, 2, 3, 1])
        result = classifier(images_out).cpu()
        change_prob = F.softmax(result, dim=1).numpy()[0, class_index]
    return images_out, change_prob


def generate_images_given_dlatent(
        dlatent: np.ndarray,
        label,
        additional_data_tuple,
        generator,
        classifier,
        class_index: int,
        sindex: int,
        s_style_min: float,
        s_style_max: float,
        style_direction_index: int,
        font_file,
        shift_size: float = 2,
        label_size: int = 2,
        draw_results_on_image: bool = True,
        resolution: int = 256,
):
    """Modifies an image given the dlatent on a specific S-index.

    Args:
        dlatent: The image dlatent, with sape [dlatent_size].
        generator: The generator model. Either StyleGAN or GLO.
        classifier: The classifier to visualize.
        class_index: The index of the class to visualize.
        sindex: The specific style index to visualize.
        s_style_min: The minimal value of the style index.
        s_style_max: The maximal value of the style index.
        style_direction_index: If 0 move s to it's min value otherwise to it's max
        value.
        font_file: A path to the font file for writing the probability on the image.
        shift_size: Factor of the shift of the style vector.
        label_size: The size of the label.
        draw_results_on_image: Whether to draw the classifier outputs on the images.

    Returns:
        The image before and after the style index modification, and the outputs of
        the classifier before and after the
        modification.
    """
    _, _, device = additional_data_tuple
    result_image = torch.zeros((resolution, 2 * resolution, 3))
    images_out = generate_image_dlatent(generator, dlatent, label, device)
    result = classifier(images_out).cpu()

    base_prob = F.softmax(result, dim=1).numpy()[0, class_index]
    if draw_results_on_image:
        result_image[:, :resolution, :] = draw_on_image(
            images_out[0], base_prob, font_file)

    change_image, change_prob = (
        generate_change_image_given_dlatent(dlatent, label, additional_data_tuple, generator, classifier,
                                            class_index, sindex,
                                            s_style_min, s_style_max,
                                            style_direction_index, shift_size,
                                            label_size))
    if draw_results_on_image:
        result_image[:, resolution:, :] = draw_on_image(
            change_image[0], change_prob, font_file)

    return result_image, change_prob, base_prob


def visualize_style_by_distance_in_s(
        generator,
        classifier,
        all_dlatents,
        all_labels,
        additional_data_tuple,
        all_style_vectors_distances: np.ndarray,
        style_min: np.ndarray,
        style_max: np.ndarray,
        sindex: int,
        style_sign_index: int,
        max_images: int,
        shift_size: float,
        font_file: str,
        label_size: int = 2,
        class_index: int = 0,
        draw_results_on_image: bool = True,
        effect_threshold: float = 0.1):
    """Returns an image visualizing the effect of a specific S-index.

    Args:
        generator: The generator model. Either StyleGAN or GLO.
        classifier: The classifier to visualize.
        all_dlatents: An array with shape [num_images, dlatent_size].
        all_style_vectors_distances: A shape of [num_images, style_size, 2].
        The distance each style from the min and max values on each image.
        style_min: The minimal value of each style, with shape [style_size].
        style_max: The maximal value of each style, with shape [style_size].
        sindex: The specific style index to visualize.
        style_sign_index: If 0 move s to its min value otherwise to its max
        value.
        max_images: Maximal number of images to visualize.
        shift_size: Factor of the shift of the style vector.
        font_file: A path to the font file for writing the probability on the image.
        label_size: The size of the label.
        class_index: The index of the class to visualize.
        draw_results_on_image: Whether to draw the classifier outputs on the images.
    """
    _, _, device = additional_data_tuple
    # Choose the dlatent indices to visualize
    images_idx = np.argsort(
        all_style_vectors_distances[:, sindex, style_sign_index])[::-1]
    if images_idx.size == 0:
        return np.array([])

    #images_idx = images_idx[:min(max_images * 10, len(images_idx))]
    dlatents = torch.tensor(all_dlatents[images_idx], requires_grad=False).float().to(device)
    labels = torch.tensor(all_labels[images_idx], requires_grad=False).to(device)
    # labels = all_labels[images_idx]
    result_images = []
    result_indices = []
    for i in range(len(images_idx)):
        cur_dlatent = dlatents[i]
        cur_label = labels[i]
        (result_image, change_prob, base_prob) = generate_images_given_dlatent(
            dlatent=cur_dlatent,
            label=cur_label,
            additional_data_tuple=additional_data_tuple,
            generator=generator,
            classifier=classifier,
            class_index=class_index,
            sindex=sindex,
            s_style_min=style_min[sindex],
            s_style_max=style_max[sindex],
            style_direction_index=style_sign_index,
            font_file=font_file,
            shift_size=shift_size,
            label_size=label_size,
            draw_results_on_image=draw_results_on_image)
        # print(change_prob, base_prob)
        # if (change_prob - base_prob) < effect_threshold:
        #     continue
        if base_prob < 0.5 and change_prob < 0.5 or \
                base_prob > 0.5 and change_prob > 0.5:
            continue
        result_images.append(result_image)
        result_indices.append(i)

    if len(result_images) < 3:
        # No point in returning results with very little images
        return torch.tensor([]), result_indices
    return torch.concat(result_images[:max_images], axis=0).permute(2, 0, 1), result_indices

### END Functions from paper ###
