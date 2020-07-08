import torch
import torch.nn.functional as F


def binary_cross_entropy(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    ce = F.binary_cross_entropy(pred, target, reduction="none")
    ce = torch.sum(ce, dim=1)
    ce = torch.mean(ce, dim=0)
    return ce


def mse(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mse = torch.sum((target - pred) ** 2, dim=1)
    mse = torch.mean(mse, dim=0)
    return mse


def mae(target, pred):
    target = target.reshape((target.shape[0], -1))
    pred = pred.reshape((pred.shape[0], -1))
    mad = torch.sum(torch.abs(target - pred), dim=1)
    mad = torch.mean(mad, dim=0)
    return mad


def gce(target, pred, q=1.0):
    # generalized cross-entropy loss
    assert q > 1e-6
    pred_y = torch.sum(target * pred, dim=1)
    return torch.mean((1.0 - pred_y ** q) / q, dim=0)


def dmi(target, pred):
    # L_DMI of https://arxiv.org/pdf/1909.03388.pdf
    # mat = torch.mm(target.T, pred) / target.shape[0]  # normalizing makes the determinant too small
    mat = torch.mm(target.T, pred)
    return -torch.log(torch.abs(torch.det(mat)) + 0.001)


def fw(target, pred, T_est):
    # Forward loss function of https://arxiv.org/pdf/1609.03683.pdf.
    # The code is adapted from the original implementation.
    eps = 1e-10
    pred = torch.clamp(pred, eps, 1 - eps)
    return -torch.sum(target * torch.log(torch.mm(pred, T_est)), dim=1).mean(dim=0)


def get_classification_loss(target, pred, loss_function="ce", loss_function_param=None):
    """
    :param target: one-hot encoded vector
    :param pred: predicted logits (i.e. before softmax)
    :param loss_function: 'ce', 'mse', 'mae', 'gce', 'dmi'
    :param loss_function_param: when 'gce' this should specify the q parameter
    """
    if loss_function == "ce":
        return F.cross_entropy(input=pred, target=target.argmax(dim=1))
    if loss_function == "mse":
        return mse(target, torch.softmax(pred, dim=1))
    if loss_function == "mae":
        return mae(target, torch.softmax(pred, dim=1))
    if loss_function == "gce":
        return gce(target, torch.softmax(pred, dim=1), q=loss_function_param)
    if loss_function == "dmi":
        return dmi(target, torch.softmax(pred, dim=1))
    if loss_function == "fw":
        return fw(target, torch.softmax(pred, dim=1), T_est=loss_function_param)
    raise NotImplementedError()
