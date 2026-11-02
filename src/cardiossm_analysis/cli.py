"""Offline local analysis. Never opens a network connection or launches training."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from .metrics import patient_metrics
from .events import event_metrics
from .calibration import offset_hr
from .bootstrap import paired_interval
from .scales import scale_metrics
from .features import heldout_screen
from .temporal import rolling_average


def write_csv(path, rows):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def score(args):
    rows = []
    paths = sorted(Path(args.input).glob("*.npz"))
    if not paths:
        raise ValueError("input directory contains no patient NPZ files")
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            t, y, test, read = [z[k] for k in ("time_s", "reference", "test_mask", "reading_mask")]
            names = z["model_names"].tolist()
            pred = z["neural_predictions" if args.assemble_offset_hr else "predictions"]
            if pred.shape != (len(t), len(names)) or len(names) != len(set(names)):
                raise ValueError("model names must be unique and predictions time-by-model")
            if args.assemble_offset_hr:
                if "OffsetHR" in names:
                    raise ValueError("OffsetHR is reserved for the generated matched baseline")
                inputs = [np.zeros(len(t))] + [rolling_average(pred[:, j], t) for j in range(len(names))]
                names = ["OffsetHR"] + names
                ym = rolling_average(y, t)
                h = z['hr_bpm'] - rolling_average(z['hr_bpm'], t)
                pred = np.column_stack([offset_hr(t, y, ym, u, h,
                    z["calibration_mask"], read, qa=args.qa)[1] for u in inputs])
            for j, name in enumerate(names):
                row = dict(subject=path.stem, model=name, **patient_metrics(t, y, pred[:, j], test, read,
                    concordance_exclude_readings=args.concordance_no_reading))
                for direction in ("drop", "rise"):
                    ev = event_metrics(t, y, pred[:, j], test, read, direction=direction)
                    row.update({direction + "_" + k: v for k, v in ev.items() if k != "direction"})
                rows.append(row)
    write_csv(args.output, rows)


def compare(args):
    with Path(args.input).open() as f:
        rows = list(csv.DictReader(f))
    by = {}
    for row in rows:
        key = (row["subject"], row["model"])
        if key in by:
            raise ValueError("duplicate subject/model; split conditions and seeds before comparison")
        by[key] = row
    models = sorted({m for _, m in by})
    if args.reference not in models:
        raise ValueError("reference model absent")
    subjects = sorted(s for s, m in by if m == args.reference)
    out = []
    metric = "z_seg" if args.metric == "r_seg" else args.metric
    aggregate = "auroc" if metric.endswith("_auroc") else "average_precision" if metric.endswith("_average_precision") else metric
    for model in models:
        if model == args.reference:
            continue
        if sorted(s for s, m in by if m == model) != subjects:
            raise ValueError("model and baseline patient coverage differ")
        first = [float(by[s, model][metric]) for s in subjects]
        second = [float(by[s, args.reference][metric]) for s in subjects]
        out.append(dict(model=model, reference=args.reference, metric=args.metric,
                        **paired_interval(first, second, aggregate, args.resamples, args.seed)))
    write_csv(args.output, out)


def scales(args):
    rows = []
    for path in sorted(Path(args.input).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            for j, name in enumerate(z["model_names"].tolist()):
                records = scale_metrics(z["time_s"], z["reference"], z["predictions"][:, j], z["test_mask"], args.seconds)
                rows.extend(dict(subject=path.stem, model=name, **r) for r in records)
    write_csv(args.output, rows)


def features(args):
    rows = []
    for path in sorted(Path(args.input).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            result = heldout_screen(z["time_s"], z["reference"], z["features"], null_resamples=args.resamples, seed=args.seed)
            j = result["selected_feature"]
            if j is not None:
                result["selected_feature_name"] = str(z["feature_names"][j])
            rows.append(dict(subject=path.stem, **result))
    write_csv(args.output, rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("score", help="Patient metrics and reading-free event ranking")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--assemble-offset-hr", action="store_true")
    p.add_argument("--qa", type=float, default=2.)
    p.add_argument("--concordance-no-reading", action="store_true"); p.set_defaults(run=score)
    p = commands.add_parser("compare", help="Paired patient bootstrap with fixed tuning")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--reference", default="OffsetHR")
    p.add_argument("--metric", choices=["mae", "r_test", "r_seg", "concordance", "drop_auroc", "rise_auroc",
        "drop_average_precision", "rise_average_precision"], default="r_seg")
    p.add_argument("--resamples", type=int, default=10000); p.add_argument("--seed", type=int, default=0); p.set_defaults(run=compare)
    p = commands.add_parser("scales", help="Output averaging scale on native-resolution observations")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seconds", nargs="+", type=float, required=True); p.set_defaults(run=scales)
    p = commands.add_parser("features", help="Temporally separated feature association and max-null")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--resamples", type=int, default=499); p.add_argument("--seed", type=int, default=0); p.set_defaults(run=features)
    args = parser.parse_args(); args.run(args)


if __name__ == "__main__":
    main()
