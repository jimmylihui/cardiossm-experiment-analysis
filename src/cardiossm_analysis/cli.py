"""Offline local analysis. Never opens a network connection or launches training."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from .metrics import patient_metrics
from .events import event_metrics
from .calibration import assemble_cohort
from .bootstrap import paired_interval, candidate_interval, holm_adjust
from .adapter import align_training_outputs
from .scales import scale_metrics
from .features import heldout_screen
from .temporal import rolling_average
from .runs import record_runs, RUN_GAP_BASES


def write_csv(path, rows):
    target = external_output(path); target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader(); writer.writerows(rows)


def external_output(path):
    target=Path(path).resolve()
    if target.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('derived outputs must stay outside the code repository')
    return target


def prepare(args):
    directories={}
    for item in args.models:
        name,separator,path=item.partition('=')
        if not separator or not name or name in directories:raise ValueError('use unique NAME=external-directory model inputs')
        directories[name]=Path(path)
    files={}
    for name,directory in directories.items():
        paths=list(directory.rglob('*.npz'))
        if len(paths)!=len({p.stem for p in paths}):raise ValueError('duplicate patient outputs across folds')
        files[name]={p.stem:p for p in paths}
    subjects=sorted(next(iter(files.values())))
    if not subjects or any(sorted(v)!=subjects for v in files.values()):raise ValueError('model patient coverage differs')
    output=external_output(args.output);output.mkdir(parents=True,exist_ok=False)
    for subject in subjects:
        records={}
        for name,paths in files.items():
            with np.load(paths[subject],allow_pickle=False) as z:records[name]={k:z[k] for k in z.files}
        np.savez_compressed(output/(subject+'.npz'),**align_training_outputs(records,args.tag))


def metric_row(subject,name,r,p,args):
    t,y,test,read=[r[k] for k in ('time_s','reference','test_mask','reading_mask')]
    runs,basis=record_runs(r)
    block=args.network_scale=='block'
    scale='block_2min' if block else 'past_only_60min'
    row=dict(subject=subject,model=name,**patient_metrics(t,y,p,test,read,
        concordance_exclude_readings=args.concordance_no_reading,run_id=runs,include_changes=block),
        reference_scale=scale,prediction_scale=scale,run_gap_basis=basis)
    if block:
        for direction in ('drop','rise'):
            row.update({direction+'_'+k:v for k,v in event_metrics(t,y,p,test,read,direction=direction,run_id=runs).items() if k!='direction'})
    return row


def score(args):
    rows=[];records=[];candidate_rows=[]
    paths = sorted(Path(args.input).glob("*.npz"))
    if not paths:
        raise ValueError("input directory contains no patient NPZ files")
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            records.append(dict({k:z[k] for k in z.files},subject=path.stem))
    if args.assemble_offset_hr:
        if not args.assemble_models or args.network_only or args.network_scale!='block':
            raise ValueError('specify --assemble-models explicitly; assembly uses unsmoothed block inputs')
        records,audit=assemble_cohort(records,args.assemble_models,args.qa)
    elif args.qa is not None or args.assemble_models or args.candidate_output:
        raise ValueError('qa/model selection/candidates require assembly')
    for i,r in enumerate(records):
        runs,_=record_runs(r)
        names=r['model_names'].tolist()
        if len(names)!=len(set(names)):raise ValueError('model names must be unique')
        pred=r['neural_predictions'].copy() if args.network_only else r['predictions']
        if pred.shape!=(len(r['time_s']),len(names)):raise ValueError('predictions must be time-by-model')
        if args.network_scale=='slow':
            if not args.network_only:raise ValueError('slow network scoring requires --network-only to avoid double smoothing')
            pred=rolling_average(pred,r['time_s'],run_id=runs)
            r=dict(r,reference=rolling_average(r['reference'],r['time_s'],run_id=runs))
        for j,name in enumerate(names):
            row=metric_row(r['subject'],name,r,pred[:,j],args)
            if args.assemble_offset_hr:
                assembled=name=='OffsetHR' or name in args.assemble_models
                row.update(qa=float(r['selected_qa'][j]) if assembled else float('nan'),
                    qa_selection=r['qa_selection'] if assembled else 'network_output',tuning_patients=len(records)-1)
                if args.candidate_output:
                    for c,q in enumerate(audit['qa_candidates']):
                        candidate_rows.append(dict(metric_row(r['subject'],name,r,audit['candidate_predictions'][i][:,j,c],args),
                            qa_candidate=float(q),selection_error=float(audit['candidate_errors'][i,j,c])))
            rows.append(row)
    write_csv(args.output, rows)
    if args.candidate_output:write_csv(args.candidate_output,candidate_rows)


def compare(args):
    with Path(args.input).open() as f:
        rows = list(csv.DictReader(f))
    metadata={}
    for field in ('reference_scale','prediction_scale','run_gap_basis'):
        values={r.get(field) for r in rows}
        if len(values)!=1 or None in values or '' in values:
            raise ValueError('comparisons require one explicit matching '+field+'; regenerate legacy score files')
        metadata[field]=next(iter(values))
    if metadata['reference_scale'] not in ('block_2min','past_only_60min') or metadata['prediction_scale']!=metadata['reference_scale']:
        raise ValueError('reference and prediction must declare the same supported scale')
    if metadata['run_gap_basis'] not in RUN_GAP_BASES:
        raise ValueError('unsupported comparison run_gap_basis')
    if (args.metric=='concordance' or args.metric.startswith(('drop_','rise_'))) and metadata['reference_scale']!='block_2min':
        raise ValueError('block-change metrics cannot be compared on a slow target')
    by = {};candidates={}
    for row in rows:
        key = (row["subject"], row["model"])
        if args.tuning=='reselect':
            cell=(*key,float(row['qa_candidate']))
            if cell in candidates:raise ValueError('duplicate patient/model/candidate')
            candidates[cell]=row
            by[key]=row
            continue
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
        if args.metric=='mae' and args.aggregation!='mean':raise ValueError('Methods aggregates patient MAE by mean')
        options=dict(aggregation=args.aggregation,contrast=args.contrast)
        if args.tuning=='fixed':
            first=[float(by[s,model][metric]) for s in subjects];second=[float(by[s,args.reference][metric]) for s in subjects]
            interval=paired_interval(first,second,aggregate,args.resamples,args.seed,**options)
        else:
            def matrix(name,column):
                grid=sorted({q for s,m,q in candidates if m==name})
                try:return np.array([[float(candidates[s,name,q][column]) for q in grid] for s in subjects])
                except KeyError as error:raise ValueError('candidate coverage must be complete') from error
            interval=candidate_interval(matrix(model,metric),matrix(model,'selection_error'),
                matrix(args.reference,metric),matrix(args.reference,'selection_error'),aggregate,args.resamples,args.seed,**options)
        out.append(dict(model=model,reference=args.reference,metric=args.metric,tuning=args.tuning,**metadata,**interval))
    write_csv(args.output, out)


def scales(args):
    rows = []
    for path in sorted(Path(args.input).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            runs,basis=record_runs(z)
            for j, name in enumerate(z["model_names"].tolist()):
                records = scale_metrics(z["time_s"], z["reference"], z["predictions"][:, j], z["test_mask"], args.seconds,run_id=runs)
                rows.extend(dict(subject=path.stem, model=name,run_gap_basis=basis, **r) for r in records)
    write_csv(args.output, rows)


def features(args):
    rows = []
    for path in sorted(Path(args.input).glob("*.npz")):
        with np.load(path, allow_pickle=False) as z:
            runs,basis=record_runs(z)
            result = heldout_screen(z["time_s"], z["reference"], z["features"], null_resamples=args.resamples, seed=args.seed,run_id=runs)
            j = result["selected_feature"]
            if j is not None:
                result["selected_feature_name"] = str(z["feature_names"][j])
            rows.append(dict(subject=path.stem,run_gap_basis=basis, **result))
    write_csv(args.output, rows)


def adjust_p(args):
    with Path(args.input).open() as f:rows=list(csv.DictReader(f))
    families={}
    for i,row in enumerate(rows):
        if not row.get('family'):raise ValueError('each supplied p-value needs an explicit family')
        families.setdefault(row['family'],[]).append(i)
    for ids in families.values():
        adjusted=holm_adjust([float(rows[i]['p_value']) for i in ids])
        for i,p in zip(ids,adjusted):rows[i]['holm_p']=float(p)
    write_csv(args.output,rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p=commands.add_parser('prepare',help='Align external training outputs at unsmoothed block resolution')
    p.add_argument('--models',nargs='+',required=True,help='NAME=external-directory for each model')
    p.add_argument('--tag',required=True);p.add_argument('--output',required=True);p.set_defaults(run=prepare)
    p = commands.add_parser("score", help="Patient metrics and reading-free event ranking")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--assemble-offset-hr", action="store_true")
    p.add_argument('--assemble-models',nargs='+',help='Only these explicitly named networks receive offset and HR')
    p.add_argument('--network-only',action='store_true')
    p.add_argument('--network-scale',choices=['block','slow'],default='block')
    p.add_argument('--candidate-output',help='External candidate metrics CSV for bootstrap reselection')
    p.add_argument("--qa", type=float, help='Explicit fixed sensitivity analysis; default selects on other-patient test MAE')
    p.add_argument("--concordance-no-reading", action="store_true"); p.set_defaults(run=score)
    p = commands.add_parser("compare", help="Paired patient bootstrap with fixed tuning")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--reference", default="OffsetHR")
    p.add_argument('--aggregation',choices=['mean','median'],required=True)
    p.add_argument('--contrast',choices=['difference_of_aggregates','median_paired_delta'],required=True)
    p.add_argument('--tuning',choices=['fixed','reselect'],required=True)
    p.add_argument("--metric", choices=["mae", "r_test", "r_seg", "concordance", "drop_auroc", "rise_auroc",
        "drop_average_precision", "rise_average_precision"], default="r_seg")
    p.add_argument("--resamples", type=int, default=10000); p.add_argument("--seed", type=int, default=0); p.set_defaults(run=compare)
    p = commands.add_parser("scales", help="Output averaging scale on native-resolution observations")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--seconds", nargs="+", type=float, required=True); p.set_defaults(run=scales)
    p = commands.add_parser("features", help="Temporally separated feature association and max-null")
    p.add_argument("--input", required=True); p.add_argument("--output", required=True)
    p.add_argument("--resamples", type=int, default=499); p.add_argument("--seed", type=int, default=0); p.set_defaults(run=features)
    p=commands.add_parser('adjust-p',help='Holm adjustment of explicitly supplied tests and comparison families')
    p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.set_defaults(run=adjust_p)
    args = parser.parse_args(); args.run(args)


if __name__ == "__main__":
    main()
