"""Utilities for the HAGRID YOLOv5 live demo notebook."""

from __future__ import annotations

import os
import pathlib
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from IPython.display import display
from matplotlib.ticker import MaxNLocator


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def add_bar_value_labels(ax, fmt: str = "{:.3f}", fontsize: int = 8, padding: int = 3) -> None:
    for container in ax.containers:
        labels = []
        for bar in container:
            value = bar.get_height()
            labels.append("" if np.isnan(value) else fmt.format(value))
        ax.bar_label(container, labels=labels, padding=padding, fontsize=fontsize)


def add_hist_count_labels(ax, fontsize: int = 8) -> None:
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    for patch in ax.patches:
        count = patch.get_height()
        if count <= 0:
            continue
        ax.annotate(
            f"{int(round(count))}",
            (patch.get_x() + patch.get_width() / 2, count),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            xytext=(0, 3),
            textcoords="offset points",
        )


def load_toml(path: Path) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib
    with path.open("rb") as f:
        return tomllib.load(f)


def load_demo_config(root: Path) -> dict:
    env_path = root / "colab_env.toml"
    env = load_toml(env_path) if env_path.exists() else {}
    paths = env.get("paths", {})
    return {
        "root": root,
        "image_dir": root / paths.get("default_image_dir", "datasets/HAGRID-YOLO/test/images"),
        "curated_dir": root / paths.get("curated_image_dir", "assets/curated_test_set"),
        "single_dir": root / paths.get("single_image_dir", "assets/inference_images"),
        "weights": root / paths.get("weights", "artifacts/runs/train/exp_batchsize_32/weights/best.pt"),
        "data_yaml": root / paths.get("data_yaml", "datasets/HAGRID-YOLO/data.yaml"),
        "yolov5_runtime": root / paths.get("yolov5_runtime", "yolov5_runtime"),
        "img_size": 384,
        "conf_thres": 0.50,
        "iou_thres": 0.45,
        "max_det": 100,
    }


def load_dataset_distribution(config: dict):
    dataset_root = config["data_yaml"].parent
    with config["data_yaml"].open("r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)
    class_names = data_cfg.get("names", [])
    split_map = {"train": "Train", "valid": "Validation", "test": "Test"}

    image_rows = []
    class_rows = []
    for split_dir, split_label in split_map.items():
        image_dir = dataset_root / split_dir / "images"
        label_dir = dataset_root / split_dir / "labels"
        images = list_images(image_dir) if image_dir.exists() else []
        labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
        # Fall back to label count when images folder is absent (e.g. train/valid stripped from zip)
        image_count = len(images) if images else len(labels)
        image_rows.append({"split": split_label, "images": image_count, "labels": len(labels)})

        class_counts = {idx: 0 for idx in range(len(class_names))}
        total_instances = 0
        for label_path in labels:
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(float(parts[0]))
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                total_instances += 1
        for class_id, count in class_counts.items():
            class_rows.append(
                {
                    "split": split_label,
                    "class_id": class_id,
                    "class_name": class_names[class_id] if class_id < len(class_names) else str(class_id),
                    "instances": count,
                    "total_instances": total_instances,
                }
            )

    image_df = pd.DataFrame(image_rows)
    image_df["image_ratio"] = image_df["images"] / image_df["images"].sum()
    class_df = pd.DataFrame(class_rows)
    class_df["instance_ratio_in_split"] = class_df["instances"] / class_df["total_instances"].replace(0, np.nan)
    return image_df, class_df


def show_dataset_distribution(image_df: pd.DataFrame, class_df: pd.DataFrame) -> None:
    plt.close("all")
    display(image_df.assign(image_ratio_pct=(image_df["image_ratio"] * 100).round(2)))

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    sns.barplot(data=image_df, x="split", y="images", ax=axes[0])
    axes[0].set_title("Dataset Split Image Counts")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Images")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%d")

    ratio_df = image_df.copy()
    ratio_df["image_ratio_pct"] = ratio_df["image_ratio"] * 100
    sns.barplot(data=ratio_df, x="split", y="image_ratio_pct", ax=axes[1])
    axes[1].set_title("Dataset Split Ratio")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Ratio (%)")
    axes[1].set_ylim(0, max(100, ratio_df["image_ratio_pct"].max() + 5))
    for container in axes[1].containers:
        axes[1].bar_label(container, fmt="%.1f%%")

    fig.tight_layout()
    display(fig)
    plt.close(fig)

    display(class_df.pivot_table(index="class_name", columns="split", values="instances", aggfunc="sum").fillna(0).astype(int))
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    sns.barplot(data=class_df, x="class_name", y="instances", hue="split", ax=ax)
    ax.set_title("Class Instance Distribution by Split")
    ax.set_xlabel("Class")
    ax.set_ylabel("Instances")
    ax.legend(title="Split")
    fig.tight_layout()
    display(fig)
    plt.close(fig)


def prepare_yolov5_imports(root: Path, runtime_root: Path | None = None):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".ultralytics"))
    runtime_root = runtime_root or (root / "yolov5_runtime")
    for path in (root, runtime_root):
        if str(path) not in sys.path:
            sys.path.append(str(path))
    if os.name != "nt":
        pathlib.WindowsPath = pathlib.PosixPath

    from models.common import DetectMultiBackend
    from utils.augmentations import letterbox
    from utils.general import check_img_size, non_max_suppression, scale_boxes
    from utils.torch_utils import select_device

    return DetectMultiBackend, letterbox, check_img_size, non_max_suppression, scale_boxes, select_device


def load_model(config: dict):
    DetectMultiBackend, letterbox, check_img_size, nms, scale_boxes, select_device = prepare_yolov5_imports(
        config["root"], config["yolov5_runtime"]
    )
    device = select_device("")
    model = DetectMultiBackend(str(config["weights"]), device=device, data=str(config["data_yaml"]), fp16=False)
    imgsz = check_img_size((config["img_size"], config["img_size"]), s=model.stride)
    model.warmup(imgsz=(1, 3, imgsz[0], imgsz[1]))
    return {
        "model": model,
        "device": device,
        "imgsz": imgsz,
        "letterbox": letterbox,
        "nms": nms,
        "scale_boxes": scale_boxes,
        "names": model.names,
        "stride": model.stride,
        "pt": model.pt,
    }


def list_images(folder: Path) -> List[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def preprocess_image(im0_bgr: np.ndarray, runtime: dict) -> torch.Tensor:
    im = runtime["letterbox"](im0_bgr, runtime["imgsz"], stride=runtime["stride"], auto=runtime["pt"])[0]
    im = im[:, :, ::-1].transpose(2, 0, 1)
    im = np.ascontiguousarray(im)
    im = torch.from_numpy(im).to(runtime["device"]).float() / 255.0
    return im[None] if im.ndimension() == 3 else im


def class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(class_id + 17)
    return tuple(int(x) for x in rng.integers(40, 240, size=3).tolist())


def annotate_image(im0_bgr: np.ndarray, detections: List[Dict]) -> np.ndarray:
    annotated = im0_bgr.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["xyxy"]]
        color = class_color(int(det["class_id"]))
        label = f"{det['class_name']} {det['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)


def predict(image_path: Path, runtime: dict, config: dict) -> Dict:
    image_path = Path(image_path)
    timestamps: Dict[str, float] = {}

    timestamps["read_start"] = time.perf_counter()
    im0_bgr = cv2.imread(str(image_path))
    timestamps["read_end"] = time.perf_counter()
    if im0_bgr is None:
        raise ValueError(f"Failed to read image: {image_path}")

    timestamps["preprocess_start"] = time.perf_counter()
    im = preprocess_image(im0_bgr, runtime)
    timestamps["preprocess_end"] = time.perf_counter()

    timestamps["inference_start"] = time.perf_counter()
    with torch.no_grad():
        pred = runtime["model"](im, augment=False, visualize=False)
    if runtime["device"].type != "cpu":
        torch.cuda.synchronize()
    timestamps["inference_end"] = time.perf_counter()

    timestamps["postprocess_start"] = time.perf_counter()
    pred = runtime["nms"](pred, config["conf_thres"], config["iou_thres"], max_det=config["max_det"])
    detections: List[Dict] = []
    det_tensor = pred[0]
    if len(det_tensor):
        det_tensor[:, :4] = runtime["scale_boxes"](im.shape[2:], det_tensor[:, :4], im0_bgr.shape).round()
        for *xyxy, conf, cls in det_tensor.cpu().numpy().tolist():
            cls_id = int(cls)
            names = runtime["names"]
            detections.append(
                {
                    "xyxy": xyxy,
                    "confidence": float(conf),
                    "class_id": cls_id,
                    "class_name": names[cls_id] if isinstance(names, list) else names.get(cls_id, str(cls_id)),
                }
            )
    timestamps["postprocess_end"] = time.perf_counter()

    timestamps["annotate_start"] = time.perf_counter()
    annotated_rgb = annotate_image(im0_bgr, detections)
    timestamps["annotate_end"] = time.perf_counter()

    return {
        "image_path": image_path,
        "image_name": image_path.name,
        "width": int(im0_bgr.shape[1]),
        "height": int(im0_bgr.shape[0]),
        "detections": detections,
        "num_detections": len(detections),
        "annotated_rgb": annotated_rgb,
        "timestamps": timestamps,
    }


def timing_record(result: Dict) -> Dict:
    ts = result["timestamps"]
    return {
        "image_name": result["image_name"],
        "width": result["width"],
        "height": result["height"],
        "num_detections": result["num_detections"],
        "read_ms": (ts["read_end"] - ts["read_start"]) * 1000,
        "preprocess_ms": (ts["preprocess_end"] - ts["preprocess_start"]) * 1000,
        "inference_ms": (ts["inference_end"] - ts["inference_start"]) * 1000,
        "postprocess_ms": (ts["postprocess_end"] - ts["postprocess_start"]) * 1000,
        "annotate_ms": (ts["annotate_end"] - ts["annotate_start"]) * 1000,
        "total_ms": (ts["annotate_end"] - ts["read_start"]) * 1000,
    }


def predict_folder(folder: Path, runtime: dict, config: dict, limit: int | None = None):
    paths = list_images(folder)
    if limit:
        paths = paths[:limit]
    results = [predict(path, runtime, config) for path in paths]
    table = pd.DataFrame([timing_record(result) for result in results])
    return results, table


def show_single_result(result: Dict, title: str = "Single Image Inference") -> None:
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.imshow(result["annotated_rgb"])
    ax.set_title(f"{title}: {result['image_name']} | Detections: {result['num_detections']}")
    ax.axis("off")
    display(fig)
    plt.close(fig)


def show_prediction_grid(results: List[Dict], rows: int = 4, cols: int = 4, pages: int = 2) -> None:
    plt.close("all")
    per_page = rows * cols
    max_images = per_page * pages
    shown_results = results[:max_images]
    if len(results) > max_images:
        print(f"Showing first {max_images} of {len(results)} images.")

    for page_idx in range(pages):
        page_results = shown_results[page_idx * per_page : (page_idx + 1) * per_page]
        if not page_results and page_idx > 0:
            break
        fig, axes = plt.subplots(rows, cols, figsize=(18, 18))
        for ax_idx, ax in enumerate(axes.flatten()):
            if ax_idx < len(page_results):
                item = page_results[ax_idx]
                ax.imshow(item["annotated_rgb"])
                ax.set_title(f"{item['image_name']}\nDetections: {item['num_detections']}", fontsize=9)
            ax.axis("off")
        fig.suptitle(f"Prediction Grid Page {page_idx + 1} - {rows}x{cols}", fontsize=18)
        plt.tight_layout()
        display(fig)
        plt.close(fig)


def show_latency_summary(inference_df: pd.DataFrame) -> pd.DataFrame:
    plt.close("all")
    time_cols = ["read_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "annotate_ms", "total_ms"]
    summary = inference_df[time_cols].agg(["mean", "median", "min", "max"]).T
    summary["p95"] = inference_df[time_cols].quantile(0.95)
    summary["fps_from_mean"] = 1000 / summary["mean"]
    display(summary.round(3))

    stage_cols = ["read_ms", "preprocess_ms", "inference_ms", "postprocess_ms", "annotate_ms"]
    stage_df = inference_df[stage_cols].melt(var_name="stage", value_name="latency_ms")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    sns.barplot(data=stage_df, x="stage", y="latency_ms", estimator="mean", errorbar=None, ax=axes[0])
    axes[0].set_title("Average Latency by Pipeline Stage")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].tick_params(axis="x", rotation=25)
    add_bar_value_labels(axes[0], fmt="{:.2f}", fontsize=8)

    sns.histplot(
        data=inference_df,
        x="total_ms",
        bins=min(20, max(5, len(inference_df))),
        kde=True,
        line_kws={"linewidth": 2, "label": "KDE trend"},
        ax=axes[1],
    )
    axes[1].set_title("Total Latency Distribution (Histogram + KDE Trend)")
    axes[1].set_xlabel("Total Latency (ms)")
    axes[1].set_ylabel("Count (images)")
    add_hist_count_labels(axes[1])
    if axes[1].lines:
        axes[1].legend()
    fig.tight_layout()
    display(fig)
    plt.close(fig)
    return summary


def experiment_metadata(run_name: str, opt: dict) -> Dict:
    hyp = opt.get("hyp", {})
    lr0 = hyp.get("lr0") if isinstance(hyp, dict) else None
    batch_size = opt.get("batch_size")

    if run_name.startswith("exp_batchsize_"):
        value = run_name.replace("exp_batchsize_", "")
        return {
            "experiment_group": "Batch Size",
            "experiment_label": f"Batch {value}",
            "experiment_value": int(value),
            "lr0": lr0,
        }
    if run_name.startswith("exp_lr_"):
        value = run_name.replace("exp_lr_", "").replace("p", ".")
        return {
            "experiment_group": "Learning Rate",
            "experiment_label": f"LR {value}",
            "experiment_value": float(value),
            "lr0": lr0,
        }
    return {
        "experiment_group": "Baseline / Other",
        "experiment_label": run_name,
        "experiment_value": batch_size if batch_size is not None else 0,
        "lr0": lr0,
    }


def load_runs(root: Path):
    run_rows = []
    curve_rows = []
    runs_train_dir = root / "artifacts" / "runs" / "train"
    if not runs_train_dir.exists():
        runs_train_dir = root / "runs" / "train"
    if not runs_train_dir.exists():
        return pd.DataFrame(), pd.DataFrame()
    for run_dir in sorted(p for p in runs_train_dir.iterdir() if p.is_dir()):
        results_csv = run_dir / "results.csv"
        opt_yaml = run_dir / "opt.yaml"
        if not results_csv.exists() or not opt_yaml.exists():
            continue
        results = pd.read_csv(results_csv)
        results.columns = [c.strip() for c in results.columns]
        with opt_yaml.open("r", encoding="utf-8") as f:
            opt = yaml.safe_load(f)
        best = results.loc[results["metrics/mAP_0.5:0.95"].astype(float).idxmax()]
        last = results.iloc[-1]
        metadata = experiment_metadata(run_dir.name, opt)
        run_rows.append(
            {
                "run": run_dir.name,
                **metadata,
                "epochs_config": opt.get("epochs"),
                "epochs_completed": len(results),
                "batch_size": opt.get("batch_size"),
                "imgsz": opt.get("imgsz"),
                "weights": opt.get("weights"),
                "best_epoch": int(best["epoch"]),
                "precision": float(best["metrics/precision"]),
                "recall": float(best["metrics/recall"]),
                "mAP50": float(best["metrics/mAP_0.5"]),
                "mAP50_95": float(best["metrics/mAP_0.5:0.95"]),
                "last_train_box_loss": float(last["train/box_loss"]),
                "last_val_box_loss": float(last["val/box_loss"]),
            }
        )
        tmp = results.copy()
        tmp["run"] = run_dir.name
        tmp["experiment_group"] = metadata["experiment_group"]
        tmp["experiment_label"] = metadata["experiment_label"]
        curve_rows.append(tmp)
    runs_df = pd.DataFrame(run_rows)
    if not runs_df.empty:
        group_order = {"Batch Size": 0, "Learning Rate": 1, "Baseline / Other": 2}
        runs_df["group_order"] = runs_df["experiment_group"].map(group_order).fillna(99)
        runs_df = runs_df.sort_values(["group_order", "experiment_value", "run"]).drop(columns=["group_order"])
    curves_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()
    return runs_df, curves_df


def show_run_comparison(runs_df: pd.DataFrame, curves_df: pd.DataFrame) -> None:
    plt.close("all")
    display(runs_df)
    if runs_df.empty:
        return

    metrics = ["precision", "recall", "mAP50", "mAP50_95"]
    plot_groups = ["Batch Size", "Learning Rate", "Baseline / Other"]
    for group in plot_groups:
        subset = runs_df[runs_df["experiment_group"] == group].copy()
        if subset.empty:
            continue
        metric_plot_df = subset.melt(
            id_vars=["run", "experiment_label", "epochs_config", "batch_size", "lr0", "imgsz"],
            value_vars=metrics,
            var_name="metric",
            value_name="score",
        )
        fig, ax = plt.subplots(1, 1, figsize=(max(10, len(subset) * 2.4), 5.5))
        sns.barplot(data=metric_plot_df, x="experiment_label", y="score", hue="metric", ax=ax)
        ax.set_title(f"{group} Experiment Scores")
        ax.set_ylim(max(0, metric_plot_df["score"].min() - 0.05), 1.08)
        ax.set_xlabel("Experiment")
        ax.set_ylabel("Validation Score")
        ax.legend(title="Metric", loc="lower right")
        add_bar_value_labels(ax, fmt="{:.3f}", fontsize=7, padding=2)
        ax.tick_params(axis="x", rotation=0)
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    ranking = runs_df.sort_values("mAP50_95", ascending=False)
    fig, ax = plt.subplots(1, 1, figsize=(max(10, len(ranking) * 1.4), 5))
    sns.barplot(data=ranking, x="experiment_label", y="mAP50_95", hue="experiment_group", dodge=False, ax=ax)
    ax.set_title("Best mAP@0.5:0.95 Ranking")
    ax.set_xlabel("Experiment")
    ax.set_ylabel("mAP@0.5:0.95")
    ax.set_ylim(max(0, ranking["mAP50_95"].min() - 0.05), min(1.08, ranking["mAP50_95"].max() + 0.08))
    add_bar_value_labels(ax, fmt="{:.3f}", fontsize=8, padding=3)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    display(fig)
    plt.close(fig)

    if curves_df.empty:
        return

    for group in plot_groups[:2]:
        subset_curves = curves_df[curves_df["experiment_group"] == group].copy()
        if subset_curves.empty:
            continue
        fig, ax = plt.subplots(1, 1, figsize=(12, 5))
        sns.lineplot(
            data=subset_curves,
            x="epoch",
            y="metrics/mAP_0.5:0.95",
            hue="experiment_label",
            marker="o",
            ax=ax,
        )
        ax.set_title(f"{group} mAP@0.5:0.95 over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("mAP@0.5:0.95")
        fig.tight_layout()
        display(fig)
        plt.close(fig)

    loss_plot_df = curves_df.melt(
        id_vars=["run", "epoch"],
        value_vars=["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss"],
        var_name="loss",
        value_name="value",
    )
    g = sns.relplot(
        data=loss_plot_df,
        x="epoch",
        y="value",
        hue="run",
        col="loss",
        kind="line",
        marker="o",
        col_wrap=2,
        height=4,
        facet_kws={"sharey": False},
    )
    g.fig.suptitle("Training and Validation Loss Curves", y=1.03)
    display(g.fig)
    plt.close(g.fig)
