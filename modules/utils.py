from torch.utils.data import Subset, DataLoader
import methods
from collections import defaultdict
from tqdm import tqdm
import torch
import os
import re


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def to_numpy(x):
    if x.requires_grad:
        x = x.detach()
    if x.device.type != "cpu":
        x = x.cpu()
    return x.numpy()


def to_tensor(x, device="cpu", dtype=torch.float):
    if isinstance(x, torch.Tensor):
        x = x.to(device)
    else:
        x = torch.tensor(x, dtype=dtype, device=device)
    return x


def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.cpu()
    return [to_cpu(xt) for xt in x]


def set_requires_grad(model, value):
    for p in model.parameters():
        p.requires_grad = value


def save(model, path, optimizer=None, scheduler=None):
    print("Saving the model into {}".format(path))
    make_path(os.path.dirname(path))

    save_dict = dict()
    save_dict["model"] = model.state_dict()
    save_dict["args"] = model.args

    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()

    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()

    torch.save(save_dict, path)


def load(path, device=None, verbose=False, remove_load_from=False, model=None):
    print("Loading the model from {}".format(path))
    saved_dict = torch.load(path, map_location=device)
    args = saved_dict["args"]
    if device is not None:
        args["device"] = device

    if remove_load_from and ("load_from" in args):
        args["load_from"] = None

    if model is None:
        model_class = getattr(methods, args["class"])
        model = model_class(**args)

    if verbose:
        print(model)

    model.load_state_dict(saved_dict["model"])
    model.eval()
    return model


def apply_on_dataset(
    model,
    dataset,
    batch_size=256,
    cpu=True,
    description="",
    output_keys_regexp=".*",
    max_num_examples=2 ** 30,
    num_workers=0,
    **kwargs
):
    model.eval()
    if num_workers > 0:
        torch.multiprocessing.set_sharing_strategy("file_system")
        torch.multiprocessing.set_start_method("spawn", force=True)

    n_examples = min(len(dataset), max_num_examples)
    loader = DataLoader(
        dataset=Subset(dataset, range(n_examples)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    outputs = defaultdict(list)

    for inputs_batch, labels_batch in tqdm(loader, desc=description):
        if isinstance(inputs_batch, torch.Tensor):
            inputs_batch = [inputs_batch]
        outputs_batch = model.forward(inputs=inputs_batch, **kwargs)
        for k, v in outputs_batch.items():
            if re.fullmatch(output_keys_regexp, k) is None:
                continue
            if cpu:
                v = to_cpu(v)
            outputs[k].append(v)

        # add labels if requested
        if re.fullmatch(output_keys_regexp, "label") is not None:
            outputs["label"].append(labels_batch)

    for k in outputs:
        outputs[k] = torch.cat(outputs[k], dim=0)
        assert len(outputs[k]) == n_examples

    return outputs


def decode(model, z, to_cpu=True, batch_size=256, show_progress=False):
    """ Decode latent factors to recover inputs. """
    model.eval()
    with torch.no_grad():
        x = []
        for i in tqdm(range(0, len(z), batch_size), disable=not show_progress):
            z_batch = to_tensor(z[i : i + batch_size], model.device)
            cur_x = model.decoder(z_batch)
            if model.device != "cpu" and to_cpu:
                cur_x = cur_x.cpu()
            x.append(cur_x)
    x = torch.cat(x, dim=0)
    assert x.shape[0] == len(z)
    return x
