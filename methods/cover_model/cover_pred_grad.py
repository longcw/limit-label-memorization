from modules import nn_utils, losses, utils, baseline_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from methods.predict import PredictGradBaseClassifier
from modules import visualization as vis
from typing import List

from . import resnet


class CoverModelPredGrad(PredictGradBaseClassifier):
    """ Trains the classifier using predicted gradients. Only the output gradients are predicted.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        device="cuda",
        grad_weight_decay=0.0,
        grad_l1_penalty=0.0,
        lamb=1.0,
        sample_from_q=False,
        q_dist="Gaussian",
        loss_function="ce",
        detach=True,
        load_from=None,
        warm_up=0,
        **kwargs
    ):
        super(CoverModelPredGrad, self).__init__(**kwargs)

        self.args = {
            "num_classes": num_classes,
            "pretrained": pretrained,
            "device": device,
            "grad_weight_decay": grad_weight_decay,
            "grad_l1_penalty": grad_l1_penalty,
            "lamb": lamb,
            "sample_from_q": sample_from_q,
            "q_dist": q_dist,
            "loss_function": loss_function,
            "detach": detach,
            "load_from": load_from,
            "warm_up": warm_up,
            "class": "PredictGradOutput",
        }

        self.device = device
        self.grad_weight_decay = grad_weight_decay
        self.grad_l1_penalty = grad_l1_penalty
        self.lamb = lamb
        self.sample_from_q = sample_from_q
        self.q_dist = q_dist
        self.detach = detach
        self.loss_function = loss_function
        self.load_from = load_from
        self.warm_up = warm_up

        # lamb is the coefficient in front of the H(p,q) term. It controls the variance of predicted gradients.
        if self.q_dist == "Gaussian":
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q,
                standard_dev=np.sqrt(1.0 / 2.0 / (self.lamb + 1e-12)),
                q_dist=self.q_dist,
            )
        elif self.q_dist == "Laplace":
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=self.sample_from_q,
                standard_dev=np.sqrt(2.0) / (self.lamb + 1e-6),
                q_dist=self.q_dist,
            )
        elif self.q_dist == "dot":
            assert not self.sample_from_q
            self.grad_replacement_class = nn_utils.get_grad_replacement_class(
                sample=False
            )
        else:
            raise NotImplementedError()

        # initialize the network
        feaure_channels = 2048  # 2048 for resnet50, 512 for resnet18
        self.classifier = nn.ModuleDict(
            {
                "backbone": resnet.resnet50(pretrained),
                "fc": nn.Linear(feaure_channels, num_classes, bias=False),
            }
        )
        self.classifier = self.classifier.to(self.device)
        self.num_classes = num_classes

        # q net
        self.q_network = nn.ModuleDict(
            {
                "backbone": resnet.resnet34(pretrained=False),
                "fc": nn.Linear(512, num_classes, bias=False),
            }
        )
        self.q_network = self.q_network.to(self.device)

        if self.load_from is not None:
            print("Loading the gradient predictor model from {}".format(load_from))
            stored_net = utils.load(load_from, device="cpu")
            stored_net_params = dict(stored_net.classifier.named_parameters())
            for key, param in self.q_network.named_parameters():
                param.data = stored_net_params[key].data.to(self.device)

        self.q_loss = None
        if self.loss_function == "none":  # predicted gradient has general form
            self.q_loss = torch.nn.Sequential(
                torch.nn.Linear(2 * self.num_classes, 128),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(128, self.num_classes),
            ).to(self.device)

    @staticmethod
    def _forward(model, frames):
        batch_size, num_frames = frames.shape[:2]
        x = frames.view(batch_size * num_frames, *frames.shape[2:])
        feat = model["backbone"](x)  # batch_size * num_frames, C, h, w
        fvec = F.adaptive_avg_pool2d(feat, (1, 1)).view(batch_size * num_frames, -1)
        # fvec = self.bn_vec(fvec)

        # fvec = torch.flatten(fvec, 1)  # batch_size * num_frames, C
        fvec = fvec.view(batch_size, num_frames, -1)
        fvec = fvec.mean(1)  # batch_size , C
        pred = model["fc"](fvec)
        return pred

    def forward(self, inputs, grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        # frame: [N, 10, 3, H, W]
        frames = inputs[0].to(self.device)
        pred = self._forward(self.classifier, frames)

        # predict the gradient wrt to logits
        q_label_pred = self._forward(self.q_network, frames)
        q_label_pred_softmax = torch.softmax(q_label_pred, dim=1)
        if self.detach:
            # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
            pred_softmax = torch.softmax(pred, dim=1).detach()
        else:
            pred_softmax = torch.softmax(pred, dim=1)
        if self.loss_function == "ce":
            grad_pred = pred_softmax - q_label_pred_softmax
        elif self.loss_function == "mae":
            grad_pred = torch.sum(q_label_pred_softmax * pred_softmax, dim=1).unsqueeze(
                dim=-1
            ) * (pred_softmax - q_label_pred_softmax)
        elif self.loss_function == "none":
            grad_pred = self.q_loss(
                torch.cat([pred_softmax, q_label_pred_softmax], dim=1)
            )
        else:
            raise NotImplementedError()

        # replace the gradients
        pred_before = pred
        pred = self.grad_replacement_class.apply(pred, grad_pred)

        out = {
            "pred": pred,
            "q_label_pred": q_label_pred,
            "grad_pred": grad_pred,
            "pred_before": pred_before,
        }

        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred_before = info["pred_before"]
        grad_pred = info["grad_pred"]
        y = labels[0].to(self.device)
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()

        # classification loss
        classifier_loss = F.cross_entropy(input=info["pred"], target=y)

        # compute grad actual
        if self.detach:
            # NOTE: we detach here too, so that the classifier is trained using the predicted gradient only
            pred_softmax = torch.softmax(pred_before.detach(), dim=1)
        else:
            pred_softmax = torch.softmax(pred_before, dim=1)
        if self.loss_function in ["ce", "none"]:
            grad_actual = pred_softmax - y_one_hot
        elif self.loss_function == "mae":
            grad_actual = torch.sum(pred_softmax * y_one_hot, dim=1).unsqueeze(
                dim=-1
            ) * (pred_softmax - y_one_hot)
        else:
            raise NotImplementedError()

        # I(g : y | x) penalty
        if self.q_dist == "Gaussian":
            info_penalty = losses.mse(grad_pred, grad_actual)
        elif self.q_dist == "Laplace":
            info_penalty = losses.mae(grad_pred, grad_actual)
        elif self.q_dist == "dot":
            # this corresponds to Taylor approximation of L(w + g_t)
            info_penalty = -torch.mean((grad_pred * grad_actual).sum(dim=1), dim=0)
        else:
            raise NotImplementedError()

        batch_losses = {"classifier": classifier_loss, "info_penalty": info_penalty}

        # add predicted gradient norm penalty
        if self.grad_weight_decay > 0:
            grad_l2_loss = self.grad_weight_decay * torch.mean(
                torch.sum(grad_pred ** 2, dim=1), dim=0
            )
            batch_losses["pred_grad_l2"] = grad_l2_loss

        if self.grad_l1_penalty > 0:
            grad_l1_loss = self.grad_l1_penalty * torch.mean(
                torch.sum(torch.abs(grad_pred), dim=1), dim=0
            )
            batch_losses["pred_grad_l1"] = grad_l1_loss

        return batch_losses, info

    def on_epoch_start(self, partition, epoch, **kwargs):
        super(CoverModelPredGrad, self).on_epoch_start(
            partition=partition, epoch=epoch, **kwargs
        )
        if partition == "train":
            requires_grad = epoch >= self.warm_up
            for param in self.classifier.parameters():
                param.requires_grad = requires_grad

    def visualize(
        self, train_loader, val_loader, tensorboard=None, epoch=None, **kwargs
    ):
        visualizations = {}

        # visualize pred
        fig, _ = vis.plot_predictions(self, train_loader, key="pred")
        visualizations["predictions/pred-train"] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key="pred")
            visualizations["predictions/pred-val"] = fig

        # visualize q_label_pred
        fig, _ = vis.plot_predictions(self, train_loader, key="q_label_pred")
        visualizations["predictions/q-label-pred-train"] = fig
        if val_loader is not None:
            fig, _ = vis.plot_predictions(self, val_loader, key="q_label_pred")
            visualizations["predictions/q-label-pred-val"] = fig

        return visualizations
