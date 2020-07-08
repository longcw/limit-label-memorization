from modules import nn_utils, losses, utils, baseline_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from methods import BaseClassifier
from typing import List

from . import resnet


class CoverModel(BaseClassifier):
    """ Standard classifier trained with cross-entropy loss.
    Has an option to work on pretrained representation of x.
    Optionally, can add noise to the gradient wrt to the output logit.
    """

    def __init__(
        self,
        num_classes=2,
        pretrained=True,
        device="cuda",
        loss_function="ce",
        add_noise=False,
        noise_type="Gaussian",
        noise_std=0.0,
        loss_function_param=None,
        load_from=None,
        **kwargs
    ):
        super(CoverModel, self).__init__(**kwargs)

        self.args = {
            "num_classes": num_classes,
            "pretrained": pretrained,
            "device": device,
            "loss_function": loss_function,
            "add_noise": add_noise,
            "noise_type": noise_type,
            "noise_std": noise_std,
            "loss_function_param": loss_function_param,
            "load_from": load_from,
            "class": "StandardClassifier",
        }

        self.device = device
        self.loss_function = loss_function
        self.add_noise = add_noise
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.loss_function_param = loss_function_param
        self.load_from = load_from
        self.num_classes = num_classes

        # initialize the network
        feaure_channels = 2048  # 2048 for resnet50, 512 for resnet18
        self.classifier = nn.ModuleDict(
            {
                "backbone": resnet.resnet50(pretrained),
                "fc": nn.Linear(feaure_channels, num_classes, bias=False),
            }
        )

        self.classifier = self.classifier.to(self.device)
        self.grad_noise_class = nn_utils.get_grad_noise_class(
            standard_dev=noise_std, q_dist=noise_type
        )

        if self.load_from is not None:
            print("Loading the classifier model from {}".format(load_from))
            stored_net = utils.load(load_from, device="cpu")
            stored_net_params = dict(stored_net.classifier.named_parameters())
            for key, param in self.classifier.named_parameters():
                param.data = stored_net_params[key].data.to(self.device)

    def on_epoch_start(self, partition, epoch, loader, **kwargs):
        super(CoverModel, self).on_epoch_start(
            partition=partition, epoch=epoch, loader=loader, **kwargs
        )

        # In case of FW model, estimate the transition matrix and pass it to the model
        if partition == "train" and epoch == 0 and self.loss_function == "fw":
            T_est = baseline_utils.estimate_transition(
                load_from=self.load_from, data_loader=loader, device=self.device
            )
            self.loss_function_param = T_est

    def forward(self, inputs: List[torch.Tensor], grad_enabled=False, **kwargs):
        torch.set_grad_enabled(grad_enabled)
        # frame: [N, 10, 3, H, W]
        frames = inputs[0].to(self.device)
        batch_size, num_frames = frames.shape[:2]
        feat = self.classifier["backbone"](
            frames.view(batch_size * num_frames, *frames.shape[2:])
        )  # batch_size * num_frames, C, h, w
        fvec = F.adaptive_avg_pool2d(feat, (1, 1)).view(batch_size * num_frames, -1)
        # fvec = self.bn_vec(fvec)

        # fvec = torch.flatten(fvec, 1)  # batch_size * num_frames, C
        fvec = fvec.view(batch_size, num_frames, -1)
        fvec = fvec.mean(1)  # batch_size , C
        pred = self.classifier["fc"](fvec)
        if self.add_noise:
            pred = self.grad_noise_class.apply(pred)

        out = {"pred": pred}
        return out

    def compute_loss(self, inputs, labels, grad_enabled, **kwargs):
        torch.set_grad_enabled(grad_enabled)

        info = self.forward(inputs=inputs, grad_enabled=grad_enabled)
        pred = info["pred"]
        y = labels[0].to(self.device)

        # classification loss
        y_one_hot = F.one_hot(y, num_classes=self.num_classes).float()
        classifier_loss = losses.get_classification_loss(
            target=y_one_hot,
            pred=pred,
            loss_function=self.loss_function,
            loss_function_param=self.loss_function_param,
        )

        batch_losses = {
            "classifier": classifier_loss,
        }

        return batch_losses, info
