import numpy as np
import torch
import torch.nn.functional as F
from basicsr.utils.registry import METRIC_REGISTRY

@METRIC_REGISTRY.register()
def calculate_climate_mae(img, img2, crop_border):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    maes = []
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]]
        target = img2[:, [c]]
        mae = F.l1_loss(input, target)
        maes.append(mae)

    return maes

@METRIC_REGISTRY.register()
def calculate_climate_mse(img, img2, crop_border):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mses = []
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]]
        target = img2[:, [c]]
        mse = F.mse_loss(input, target)
        mses.append(mse)

    return mses

@METRIC_REGISTRY.register()
def calculate_climate_rmse(img, img2, crop_border):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[..., crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[..., crop_border:-crop_border, crop_border:-crop_border]

    mses = []
    if img.shape[1] == 0:
        return [torch.tensor(0, dtype=img.dtype, device=img.device)]
    channels = img.shape[1]
    for c in range(channels):
        input = img[:, [c]]
        target = img2[:, [c]]
        mse = F.mse_loss(input, target)
        mses.append(mse)
    return mses

