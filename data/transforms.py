import random
import numpy as np

def augment(imgs, hflip=True, rotation=True, return_status=False):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).

    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.

    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
        flows (list[ndarray]: Flows to be augmented. If the input is an
            ndarray, it will be transformed to a list.
            Dimension is (h, w, 2). Default: None.
        return_status (bool): Return the status of flip and rotation.
            Default: False.

    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.

    """

    # the input image should be [c h w]
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if len(img) == 0:
            return img
        if hflip:  # horizontal
            img = np.flip(img, -1)
        if vflip:  # vertical
            img = np.flip(img, -2)
        if rot90:
            img = np.swapaxes(img, -1, -2)
        return img

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [_augment(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]

    if return_status:
        return imgs, (hflip, vflip, rot90)
    else:
        return imgs


def paired_random_crop(img_gts, img_lqs, gt_patch_size, scale):
    # the default input format is c, h, w
    h_lq, w_lq = img_lqs[0].shape[-2:]
    h_gt, w_gt = img_gts[0].shape[-2:]

    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    img_lqs_out = []
    for v in img_lqs:
        if len(v) == 0:
            img_lqs_out.append(v)
        else:
            img_lqs_out.append(v[..., top:top + lq_patch_size, left:left + lq_patch_size])

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts_out = []
    for v in img_gts:
        if len(v) == 0:
            img_gts_out.append(v)
        else:
            img_gts_out.append(v[..., top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size])

    return img_gts_out, img_lqs_out


def paired_fixed_crop(img_gts, img_lqs, gt_patch_size, scale):
    # the default input format is c, h, w
    h_lq, w_lq = img_lqs[0].shape[-2:]
    h_gt, w_gt = img_gts[0].shape[-2:]
    lq_patch_size = gt_patch_size // scale
    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                         f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
        raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                         f'({lq_patch_size}, {lq_patch_size}). ')
    top = 0
    left = 0

    img_lqs_out = []
    for v in img_lqs:
        if len(v) == 0:
            img_lqs_out.append(v)
        else:
            img_lqs_out.append(v[..., top:top + lq_patch_size, left:left + lq_patch_size])

    top_gt, left_gt = int(top * scale), int(left * scale)
    img_gts_out = []
    for v in img_gts:
        if len(v) == 0:
            img_gts_out.append(v)
        else:
            img_gts_out.append(v[..., top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size])

    return img_gts_out, img_lqs_out
