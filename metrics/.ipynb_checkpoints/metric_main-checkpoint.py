import os
import time
import json
import torch
import dnnlib

from . import metric_utils
from . import frechet_inception_distance
from . import kernel_inception_distance
from . import precision_recall
from . import perceptual_path_length
from . import inception_score

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

# ----------------------------------------------------------------------------

_metric_dict = dict()  # name => fn


def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn


def is_valid_metric(metric):
    return metric in _metric_dict


def list_valid_metrics():
    return list(_metric_dict.keys())


# ----------------------------------------------------------------------------


def calc_metric(
    metric, dataset, **kwargs
):  # See metric_utils.MetricOptions for the full list of arguments.
    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts, dataset)
    
#     import pdb; pdb.set_trace()
#     real_images = real_images[0].to(opts.device)
#     generated_images = [torch.as_tensor(img) for img in generated_images]
#     generated_images = torch.stack(generated_images).to(opts.device)

#     # normed between -1 and 1 (ish)
#     real_images = (real_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#     generated_images = (generated_images * 127.5 + 128).clamp(0, 255).to(torch.uint8)

#     results = _metric_dict[metric](
#         real=real_images, generated=generated_images, device=opts.device
#     )  # (opts)
#     del real_images, generated_images

    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results=dnnlib.EasyDict(results),
        metric=metric,
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
        num_gpus=opts.num_gpus,
    )


# ----------------------------------------------------------------------------


def report_metric(result_dict, run_dir=None, snapshot_pkl=None):
    metric = result_dict["metric"]
    assert is_valid_metric(metric)
    if run_dir is not None and snapshot_pkl is not None:
        snapshot_pkl = os.path.relpath(snapshot_pkl, run_dir)

    jsonl_line = json.dumps(
        dict(result_dict, snapshot_pkl=snapshot_pkl, timestamp=time.time())
    )
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f"metric-{metric}.jsonl"), "at") as f:
            f.write(jsonl_line + "\n")


# ----------------------------------------------------------------------------
# Primary metrics.

@register_metric
def fid50k_full(opts, dataset):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    fid = frechet_inception_distance.compute_fid(opts, max_real=None, num_gen=50000, dataset=dataset)
    return dict(fid50k_full=fid)


# @register_metric
# def fid50k_full(real, generated, device):
#     fid_fn = FrechetInceptionDistance().to(device)
#     fid_fn.update(real, real=True)
#     fid_fn.update(generated, real=False)
#     fid = fid_fn.compute().item()
#     return dict(fid50k_full=fid)


@register_metric
def kid50k_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    kid = kernel_inception_distance.compute_kid(
        opts, max_real=1000000, num_gen=50000, num_subsets=100, max_subset_size=1000
    )
    return dict(kid50k_full=kid)


@register_metric
def pr50k3_full(opts):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    precision, recall = precision_recall.compute_pr(
        opts,
        max_real=200000,
        num_gen=50000,
        nhood_size=3,
        row_batch_size=10000,
        col_batch_size=10000,
    )
    return dict(pr50k3_full_precision=precision, pr50k3_full_recall=recall)


@register_metric
def ppl2_wend(opts):
    ppl = perceptual_path_length.compute_ppl(
        opts,
        num_samples=50000,
        epsilon=1e-4,
        space="w",
        sampling="end",
        crop=False,
        batch_size=2,
    )
    return dict(ppl2_wend=ppl)


@register_metric
def is50k(opts, dataset):
    opts.dataset_kwargs.update(max_size=None, xflip=False)
    mean, std = inception_score.compute_is(opts, num_gen=50000, num_splits=10, dataset=dataset)
    return dict(is50k_mean=mean, is50k_std=std)


# @register_metric
# def is50k(real, generated, device):
#     is_fn = InceptionScore().to(device)
#     is_fn.update(generated)
#     mean, std = is_fn.compute()
#     return dict(is50k_mean=mean.item(), is50k_std=std.item())


# ----------------------------------------------------------------------------
# Legacy metrics.


@register_metric
def fid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    fid = frechet_inception_distance.compute_fid(opts, max_real=50000, num_gen=50000)
    return dict(fid50k=fid)


@register_metric
def kid50k(opts):
    opts.dataset_kwargs.update(max_size=None)
    kid = kernel_inception_distance.compute_kid(
        opts, max_real=50000, num_gen=50000, num_subsets=100, max_subset_size=1000
    )
    return dict(kid50k=kid)


@register_metric
def pr50k3(opts):
    opts.dataset_kwargs.update(max_size=None)
    precision, recall = precision_recall.compute_pr(
        opts,
        max_real=50000,
        num_gen=50000,
        nhood_size=3,
        row_batch_size=10000,
        col_batch_size=10000,
    )
    return dict(pr50k3_precision=precision, pr50k3_recall=recall)


@register_metric
def ppl_zfull(opts):
    ppl = perceptual_path_length.compute_ppl(
        opts,
        num_samples=50000,
        epsilon=1e-4,
        space="z",
        sampling="full",
        crop=True,
        batch_size=2,
    )
    return dict(ppl_zfull=ppl)


@register_metric
def ppl_wfull(opts):
    ppl = perceptual_path_length.compute_ppl(
        opts,
        num_samples=50000,
        epsilon=1e-4,
        space="w",
        sampling="full",
        crop=True,
        batch_size=2,
    )
    return dict(ppl_wfull=ppl)


@register_metric
def ppl_zend(opts):
    ppl = perceptual_path_length.compute_ppl(
        opts,
        num_samples=50000,
        epsilon=1e-4,
        space="z",
        sampling="end",
        crop=True,
        batch_size=2,
    )
    return dict(ppl_zend=ppl)


@register_metric
def ppl_wend(opts):
    ppl = perceptual_path_length.compute_ppl(
        opts,
        num_samples=50000,
        epsilon=1e-4,
        space="w",
        sampling="end",
        crop=True,
        batch_size=2,
    )
    return dict(ppl_wend=ppl)


# ----------------------------------------------------------------------------
