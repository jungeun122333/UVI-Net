import copy

import cv2
import numpy as np
import scipy.ndimage as ndimage
import skimage.morphology as morphology
from PIL import Image, ImageDraw


# min-max scaling
def min_max(ct_scan):
    scaled_ct = (ct_scan - ct_scan.min()) / (ct_scan.max() - ct_scan.min())
    return scaled_ct


# masking
def mask_fn(image):
    labels, label_nb = ndimage.label(image, structure=np.ones((3, 3, 3)))
    label_count = np.bincount(labels.ravel().astype(np.int32))
    label_count[0] = 0

    mask = labels == label_count.argmax()
    mask_new = None

    # Find contour
    for i in range(mask.shape[-1]):
        if mask[..., i].sum() == 0:
            img_large = np.zeros_like((mask[..., i][..., np.newaxis]))
        else:
            contours, hier = cv2.findContours(
                mask[..., i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            cont = max(contours, key=cv2.contourArea)

            imgcopy = Image.new("L", mask[..., 0].shape, 0)
            contour = cont[:, 0, :]
            x = contour[:, 0]
            y = contour[:, 1]
            polygon_tuple = list(zip(x, y))
            ImageDraw.Draw(imgcopy).polygon(polygon_tuple, outline=0, fill=1)
            img_large = np.array(imgcopy)[..., np.newaxis]

        if mask_new is None:
            mask_new = img_large
        else:
            mask_new = np.concatenate([mask_new, img_large], axis=-1)

    return mask_new


def bed_removal(ct_image, threshold=-500):
    ct_copy = copy.deepcopy(ct_image)
    ct_copy[ct_copy > threshold] = 1
    ct_copy[ct_copy != 1] = 0
    segmentation = copy.deepcopy(ct_copy)

    mask_new = mask_fn(segmentation)
    mask_new = mask_fn(mask_new)
    mask_new = mask_fn(mask_new)
    mask_new = morphology.dilation(mask_new, np.ones((3, 3, 3)))
    mask_new = morphology.dilation(mask_new, np.ones((3, 3, 3)))
    mask_new = morphology.dilation(mask_new, np.ones((3, 3, 3)))

    img_output = np.ones(ct_image.shape) * ct_image.min()
    img_output[mask_new == 1] = ct_image[mask_new == 1]

    return img_output


def center_crop(img, threshold=-500):
    for i_1 in range(img.shape[1]):
        moving_crop_1 = img[:, i_1, :]
        if np.sum(moving_crop_1 > threshold) != 0:
            break
    for i_2 in range(img.shape[1] - 1, 0, -1):
        moving_crop_2 = img[:, i_2, :]
        if np.sum(moving_crop_2 > threshold) != 0:
            break
    for i_3 in range(img.shape[0]):
        moving_crop_3 = img[i_3, :, :]
        if np.sum(moving_crop_3 > threshold) != 0:
            break
    for i_4 in range(img.shape[0] - 1, 0, -1):
        moving_crop_4 = img[i_4, :, :]
        if np.sum(moving_crop_4 > threshold) != 0:
            break

    crop_img = img[i_3:i_4, i_1:i_2, :]
    fix_pad_x1 = np.round((img.shape[0] - crop_img.shape[0]) // 2)
    if fix_pad_x1 < 0:
        fix_pad_x1 = 0
    fix_pad_x2 = img.shape[0] - crop_img.shape[0] - fix_pad_x1
    if fix_pad_x2 < 0:
        fix_pad_x2 = 0
    fix_pad_y1 = np.round((img.shape[1] - crop_img.shape[1]) // 2)
    if fix_pad_y1 < 0:
        fix_pad_y1 = 0
    fix_pad_y2 = img.shape[1] - crop_img.shape[1] - fix_pad_y1
    if fix_pad_y2 < 0:
        fix_pad_y2 = 0
    pad_total = ((fix_pad_x1, fix_pad_x2), (fix_pad_y1, fix_pad_y2), (0, 0))
    img_pad = np.pad(crop_img, pad_total, mode="minimum")

    return img_pad