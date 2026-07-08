import argparse
import importlib.util
import statistics
import subprocess
import sys
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
TEST_DIR = ROOT / "test"


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    shape: Tuple[int, ...]
    kwargs: Dict
    note: str = ""


@dataclass
class BenchmarkResult:
    mean_ms: Optional[float]
    median_ms: Optional[float]
    stdev_ms: Optional[float]
    reason: str = ""


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_git_version(ref):
    source = subprocess.check_output(["git", "show", f"{ref}:ssim.py"], cwd=ROOT)
    temp_dir = tempfile.TemporaryDirectory(prefix="mssim_git_")
    path = Path(temp_dir.name) / "ssim_git.py"
    path.write_bytes(source)
    module = load_module("ssim_v1_benchmark", path)
    return module, temp_dir


def make_input(shape, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.rand(shape, generator=generator)
    y = torch.rand(shape, generator=generator)
    return x, y


def time_callable(fn, warmup, iters):
    with torch.no_grad():
        for _ in range(warmup):
            fn()

        samples = []
        for _ in range(iters):
            start = time.perf_counter()
            fn()
            samples.append(time.perf_counter() - start)
    return samples


def summarize(samples):
    return BenchmarkResult(
        mean_ms=statistics.mean(samples) * 1000,
        median_ms=statistics.median(samples) * 1000,
        stdev_ms=statistics.stdev(samples) * 1000 if len(samples) > 1 else 0.0,
    )


def unsupported(reason):
    return BenchmarkResult(mean_ms=None, median_ms=None, stdev_ms=None, reason=reason)


def exception_reason(exc):
    message = str(exc)
    if message:
        return f"{type(exc).__name__}: {message}"
    return type(exc).__name__


def run_callable(fn, warmup, iters):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fn()
            return summarize(time_callable(fn, warmup, iters))
    except Exception as exc:
        return unsupported(exception_reason(exc))


def current_callable(module, case, x, y):
    kwargs = dict(case.kwargs)
    return_msssim = kwargs.pop("return_msssim", False)
    metric = module.SSIM(**kwargs).eval()

    def call():
        return metric(x, y, return_msssim=return_msssim)

    return call


def v1_callable(module, case, x, y):
    kwargs = dict(case.kwargs)
    data_dim = kwargs.pop("data_dim", 2)
    if data_dim != 2:
        raise NotImplementedError("v1.0.0 only supports 2D tensors")

    reduction = kwargs.pop("reduction", "mean")
    if reduction not in (None, "mean", "batch"):
        raise NotImplementedError("v1.0.0 supports only mean/batch reductions")
    kwargs["keep_batch_dim"] = reduction == "batch"

    return_msssim = kwargs.pop("return_msssim", False)
    kwargs["return_msssim"] = return_msssim

    window_size = kwargs.get("window_size", 11)
    if not isinstance(window_size, int):
        raise NotImplementedError("v1.0.0 supports only integer window_size")

    metric = module.SSIM(**kwargs).eval()

    def call():
        return metric(x, y)

    return call


def po_hsun_su_callable(module, case, x, y):
    kwargs = dict(case.kwargs)
    if kwargs.get("data_dim", 2) != 2:
        raise NotImplementedError("Po-Hsun-Su implementation only supports 2D tensors")
    if kwargs.get("return_msssim", False):
        raise NotImplementedError("Po-Hsun-Su implementation has no MS-SSIM")
    if kwargs.get("return_log", False):
        raise NotImplementedError("Po-Hsun-Su implementation has no return_log")
    if kwargs.get("ensemble_kernel", True) is not True or kwargs.get("padding", None) is not None:
        raise NotImplementedError("Po-Hsun-Su comparison only covers padded ensemble-kernel SSIM")

    window_size = kwargs.get("window_size", 11)
    if not isinstance(window_size, int):
        raise NotImplementedError("Po-Hsun-Su implementation supports only integer window_size")

    reduction = kwargs.get("reduction", "mean")
    if reduction not in (None, "mean", "batch"):
        raise NotImplementedError("Po-Hsun-Su implementation supports only mean/batch reductions")
    metric = module.SSIM(window_size=window_size, size_average=reduction != "batch").eval()

    def call():
        return metric(x, y)

    return call


def vainf_callable(module, case, x, y):
    kwargs = dict(case.kwargs)
    data_dim = kwargs.get("data_dim", 2)
    if data_dim not in (2, 3):
        raise NotImplementedError("VainF implementation supports 2D/3D tensors")
    if kwargs.get("return_log", False):
        raise NotImplementedError("VainF implementation has no return_log")
    if kwargs.get("L", 1) != 1:
        raise NotImplementedError("VainF adapter currently maps only L=1 to data_range=1")

    window_size = kwargs.get("window_size", 11)
    if not isinstance(window_size, int):
        raise NotImplementedError("VainF implementation supports only integer window_size")

    reduction = kwargs.get("reduction", "mean")
    if reduction not in (None, "mean", "batch"):
        raise NotImplementedError("VainF implementation supports only mean/batch reductions")

    return_msssim = kwargs.get("return_msssim", False)
    size_average = reduction != "batch"
    spatial_dims = data_dim
    channel = x.shape[1]
    if return_msssim:
        metric = module.MS_SSIM(
            data_range=1,
            size_average=size_average,
            win_size=window_size,
            channel=channel,
            spatial_dims=spatial_dims,
        ).eval()
    else:
        metric = module.SSIM(
            data_range=1,
            size_average=size_average,
            win_size=window_size,
            channel=channel,
            spatial_dims=spatial_dims,
        ).eval()

    def call():
        return metric(x, y)

    return call


def make_cases(image_size):
    # These mirror the valid computational scenarios in test_ssim.py. The tensor sizes are configurable
    # so CPU benchmarking remains practical while preserving each scenario's dimensionality and parameters.
    size_2d = (2, 1, image_size, image_size)
    size_2d_ms = (2, 1, max(image_size, 192), max(image_size, 192))
    size_1d = (2, 1, image_size)
    size_3d = (2, 1, 5, image_size, image_size)

    return [
        BenchmarkCase(
            name="test_ssim1d",
            shape=size_1d,
            kwargs={"data_dim": 1, "L": 1, "padding": None, "ensemble_kernel": True, "return_msssim": False},
        ),
        BenchmarkCase(
            name="test_ssim3d",
            shape=size_3d,
            kwargs={"data_dim": 3, "L": 1, "padding": None, "ensemble_kernel": True, "return_msssim": False},
        ),
        BenchmarkCase(
            name="test_mssim3d_split",
            shape=size_3d,
            kwargs={
                "data_dim": 3,
                "L": 1,
                "ensemble_kernel": False,
                "window_size": (3, 11, 11),
                "return_msssim": True,
            },
        ),
        BenchmarkCase(
            name="test_mssim3d_ensemble",
            shape=size_3d,
            kwargs={
                "data_dim": 3,
                "L": 1,
                "ensemble_kernel": True,
                "window_size": (3, 11, 11),
                "return_msssim": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_ssim2d_w11_mean",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 11,
                "L": 1,
                "reduction": "mean",
                "return_log": False,
                "return_msssim": False,
                "ensemble_kernel": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_ssim2d_w5_mean",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "reduction": "mean",
                "return_log": False,
                "return_msssim": False,
                "ensemble_kernel": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_ssim2d_w5_batch",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "reduction": "batch",
                "return_log": False,
                "return_msssim": False,
                "ensemble_kernel": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_ssim2d_w5_batch_log",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "reduction": "batch",
                "return_log": True,
                "return_msssim": False,
                "ensemble_kernel": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_msssim2d_w5_batch_ensemble",
            shape=size_2d_ms,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "reduction": "batch",
                "return_log": False,
                "return_msssim": True,
                "ensemble_kernel": True,
            },
        ),
        BenchmarkCase(
            name="oldversion_msssim2d_w5_batch_split",
            shape=size_2d_ms,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "reduction": "batch",
                "return_log": False,
                "return_msssim": True,
                "ensemble_kernel": False,
            },
        ),
        BenchmarkCase(
            name="pohsunsu_ssim2d_w11",
            shape=size_2d,
            kwargs={"data_dim": 2, "window_size": 11, "L": 1, "ensemble_kernel": True, "return_msssim": False},
        ),
        BenchmarkCase(
            name="pohsunsu_ssim2d_w5",
            shape=size_2d,
            kwargs={"data_dim": 2, "window_size": 5, "L": 1, "ensemble_kernel": True, "return_msssim": False},
        ),
        BenchmarkCase(
            name="vainf_ssim2d_w11_split",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 11,
                "L": 1,
                "ensemble_kernel": False,
                "padding": 0,
                "return_msssim": False,
            },
        ),
        BenchmarkCase(
            name="vainf_ssim2d_w5_split",
            shape=size_2d,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "ensemble_kernel": False,
                "padding": 0,
                "return_msssim": False,
            },
        ),
        BenchmarkCase(
            name="vainf_msssim2d_w11_split",
            shape=size_2d_ms,
            kwargs={
                "data_dim": 2,
                "window_size": 11,
                "L": 1,
                "ensemble_kernel": False,
                "padding": 0,
                "return_msssim": True,
            },
        ),
        BenchmarkCase(
            name="vainf_msssim2d_w5_split",
            shape=size_2d_ms,
            kwargs={
                "data_dim": 2,
                "window_size": 5,
                "L": 1,
                "ensemble_kernel": False,
                "padding": 0,
                "return_msssim": True,
            },
        ),
    ]


def format_cell(result):
    if result.mean_ms is None:
        return "N/A"
    return f"{result.mean_ms:.3f}"


def format_config_value(value):
    if isinstance(value, str):
        return f"`{value}`"
    return f"`{value}`"


def case_config_row(case):
    kwargs = case.kwargs
    return [
        case.name,
        format_config_value(case.shape),
        format_config_value(kwargs.get("data_dim", 2)),
        format_config_value(kwargs.get("window_size", 11)),
        format_config_value(kwargs.get("padding", None)),
        format_config_value(kwargs.get("ensemble_kernel", True)),
        format_config_value(kwargs.get("reduction", "mean")),
        format_config_value(kwargs.get("return_log", False)),
        format_config_value(kwargs.get("return_msssim", False)),
    ]


def format_markdown(case_results, warmup, iters, threads, baseline_ref, image_size):
    impl_names = ["current", baseline_ref, "po_hsun_su", "vainf"]
    lines = [
        "# CPU benchmark across test_ssim scenarios",
        "",
        f"- baseline ref: `{baseline_ref}:ssim.py`",
        f"- input image size for generated 2D tensors: `{image_size}x{image_size}`",
        f"- input image size for generated 2D MS-SSIM tensors: `{max(image_size, 192)}x{max(image_size, 192)}`",
        f"- warmup: {warmup}",
        f"- iterations: {iters}",
        f"- torch threads: {threads}",
        "- values are mean milliseconds per forward call",
        "- N/A means that implementation cannot express the same scenario/API",
        "",
        "## Scenario Configurations",
        "",
        "| case | shape | data_dim | window_size | padding | ensemble_kernel | reduction | return_log | return_msssim |",
        "| --- | --- | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for case, _results in case_results:
        lines.append("| " + " | ".join(case_config_row(case)) + " |")

    lines.extend(
        [
            "",
            "## Mean Runtime (ms)",
            "",
            "| case | " + " | ".join(impl_names) + " |",
            "| --- | " + " | ".join(["---:"] * len(impl_names)) + " |",
        ]
    )
    for case, results in case_results:
        cells = [format_cell(results[name]) for name in impl_names]
        lines.append("| " + case.name + " | " + " | ".join(cells) + " |")

    lines.extend(
        [
            "",
            "## Unsupported Details",
            "",
        ]
    )
    for case, results in case_results:
        unsupported_items = [
            f"{impl}: {result.reason}" for impl, result in results.items() if result.mean_ms is None and result.reason
        ]
        if unsupported_items:
            lines.append(f"- `{case.name}`: " + "; ".join(unsupported_items))
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark test_ssim scenarios across SSIM implementations on CPU.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20241204)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--baseline-ref", default="v1.0.0", help="Git ref used as the baseline. Defaults to v1.0.0.")
    parser.add_argument("--output", type=Path, default=None, help="Optional markdown output path.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    current_module = load_module("ssim_current_benchmark", ROOT / "ssim.py")
    v1_module, temp_dir = load_git_version(args.baseline_ref)
    po_module = load_module("po_hsun_su_benchmark", TEST_DIR / "po_hsun_su_ssim.py")
    vainf_module = load_module("vainf_benchmark", TEST_DIR / "vainf_ssim.py")

    adapters: List[Tuple[str, Callable]] = [
        ("current", lambda case, x, y: current_callable(current_module, case, x, y)),
        (args.baseline_ref, lambda case, x, y: v1_callable(v1_module, case, x, y)),
        ("po_hsun_su", lambda case, x, y: po_hsun_su_callable(po_module, case, x, y)),
        ("vainf", lambda case, x, y: vainf_callable(vainf_module, case, x, y)),
    ]

    try:
        case_results = []
        for case_idx, case in enumerate(make_cases(args.image_size)):
            x, y = make_input(case.shape, args.seed + case_idx)
            results = {}
            for impl_name, build_call in adapters:
                try:
                    call = build_call(case, x, y)
                    results[impl_name] = run_callable(call, args.warmup, args.iters)
                except Exception as exc:
                    results[impl_name] = unsupported(exception_reason(exc))
            case_results.append((case, results))

        report = format_markdown(
            case_results=case_results,
            warmup=args.warmup,
            iters=args.iters,
            threads=args.threads,
            baseline_ref=args.baseline_ref,
            image_size=args.image_size,
        )
        print(report)
        if args.output is not None:
            args.output.write_text(report + "\n", encoding="utf-8")
    finally:
        temp_dir.cleanup()


if __name__ == "__main__":
    main()
