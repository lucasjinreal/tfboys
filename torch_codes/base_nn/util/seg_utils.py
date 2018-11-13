import numpy as np
import cv2
from .get_dataset_colormap import label_to_color_image
from .get_dataset_colormap import _ADE20K, _CITYSCAPES, _MAPILLARY_VISTAS, _PASCAL


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class ** 2).reshape(n_class, n_class)
    return hist


def label_accuracy_score(label_trues, label_preds, n_class):
    """Returns accuracy score evaluation result.

      - overall accuracy
      - mean accuracy
      - mean IU
      - fwavacc
    """
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (
            hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        )
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    return acc, acc_cls, mean_iu, fwavacc


def draw_seg(img, seg, cls_color_list, is_show=False, bgr_in=False):
    """
    Show segmentation result on image.
    the seg is segmentation result which after np.argmax operation

    it's (h, w) size, very pixel value is a class index

    :param img:
    :param seg:
    :param cls_color_list:
    :param is_show:
    :param bgr_in
    :return:
    """
    img = np.asarray(img, dtype=np.float)

    mask_color = np.asarray(label_to_color_image(seg, 'pascal'), dtype=np.float)
    # add this mask on img
    # img = cv2.add(img, mask_color)
    if bgr_in:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.addWeighted(img, 0.7, mask_color, 0.7, 0)
    if is_show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return img, mask_color


def draw_seg_by_dataset(img, seg, dataset, alpha=0.5, is_show=False, bgr_in=False):
    assert dataset in [_PASCAL, _CITYSCAPES, _MAPILLARY_VISTAS, _ADE20K], 'dataset not support yet.'
    img = np.asarray(img, dtype=np.float)
    if bgr_in:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    mask_color = np.asarray(label_to_color_image(seg, dataset), dtype=np.float)

    mask = (seg != 0)
    # convert mask=(h, w) -> mask=(h, w, 3)
    mask = np.dstack((mask, mask, mask)).astype(np.float)
    mask *= alpha

    # out = np.where(mask, mask_color, img)
    out = mask_color * mask + img * (1.0 - mask)
    if is_show:
        cv2.imshow('img', img)
        cv2.waitKey(0)
    return out, mask_color