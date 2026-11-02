"""Explicit one-process training entry point; never retries a failed run."""
from pathlib import Path
import argparse, fcntl, gc, hashlib, json, os, random, sys, time, traceback
import numpy as np
import torch
from . import data


def atom(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2)); tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['cardiossm', 'bcgnet', 'fsnet'], required=True)
    p.add_argument('--input', required=True, help='external preprocessed patient NPZ directory')
    p.add_argument('--splits', required=True, help='external JSON with explicit train/val/test lists per fold')
    p.add_argument('--output', required=True, help='fresh directory, outside version control')
    p.add_argument('--device', default='cuda:0'); p.add_argument('--seed', type=int, default=0)
    p.add_argument('--folds', nargs='+', type=int); p.add_argument('--intervals', nargs='+', type=int)
    p.add_argument('--lock', help='shared lock path for all training jobs; default is output parent/worker.lock')
    p.add_argument('--smoke', action='store_true', help='reduced training budgets; label outputs as smoke, not study results')
    p.add_argument('--max-rss-gib', type=float, default=16.)
    p.add_argument('--min-host-free-gib', type=float, default=64.)
    args = p.parse_args()
    if args.seed < 0 or args.max_rss_gib <= 0 or args.min_host_free_gib < 0:
        p.error('invalid seed or resource limits')
    intervals = args.intervals or [4, 8, 12, 24]
    if any(n not in [4,8,12,24] for n in intervals) or len(intervals)!=len(set(intervals)):
        p.error('intervals must be unique choices from 4,8,12,24 hours')
    dev = torch.device(args.device)
    if dev.type not in ('cpu', 'cuda'):
        p.error('supported training devices are cpu or cuda')
    if args.model == 'cardiossm' and dev.type != 'cuda':
        p.error('the published Mamba2 implementation requires a compatible CUDA runtime')
    if dev.type == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA runtime unavailable')
    out = Path(args.output).resolve()
    if out.exists():
        p.error('output already exists; automatic resume/overwrite is disabled')
    lockpath = Path(args.lock).resolve() if args.lock else out.parent/'worker.lock'
    lockpath.parent.mkdir(parents=True, exist_ok=True)
    lock = lockpath.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        p.error('another training process holds the shared lock')
    out.mkdir()
    import psutil
    own = psutil.Process()
    def check_resource():
        rss = own.memory_info().rss
        for child in own.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        available = psutil.virtual_memory().available
        if rss > args.max_rss_gib * 2**30 or available < args.min_host_free_gib * 2**30:
            atom(out/'resource_stop.json', dict(rss_gib=rss/2**30, host_available_gib=available/2**30, time=time.time()))
            raise RuntimeError('Software resource threshold crossed; no automatic retry')
    try:
        check_resource(); data.CHECK_RESOURCE = check_resource
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
        torch.set_num_threads(2)
        if dev.type == 'cuda':
            torch.cuda.set_device(dev); torch.cuda.set_per_process_memory_fraction(.30, dev)
            torch.backends.cuda.matmul.allow_tf32 = True
        data.configure(args.input, args.splits)
        folds = args.folds if args.folds is not None else sorted(map(int, data.SPLITS))
        if args.smoke:
            folds = folds[:1]
        for f in folds:
            data.split(f)
        config = dict(model=args.model, seed=args.seed, device=str(dev), folds=folds, intervals_h=intervals,
                      smoke=args.smoke,
                      split_file_sha256=hashlib.sha256(Path(args.splits).read_bytes()).hexdigest(),
                      torch_version=torch.__version__, workers=1, automatic_retry=False,
                      initial_calibration_fraction=.3, internal_calibration_train_fraction=.8,
                      neural_reading_block_order='predict_before_neural_update', output_smoothing='past_only_60min',
                      max_rss_gib=args.max_rss_gib, min_host_free_gib=args.min_host_free_gib)
        atom(out/'run_config.json', config)
        subjects = data.load_all(require_features=args.model != 'fsnet')
        completed = []
        from . import core as trainer
        trainer.ROOT = out; trainer.SEED = args.seed; trainer.NS = intervals
        trainer.PRETRAIN_EPOCHS = 1 if args.smoke else 3
        trainer.FIT_EPOCHS = 1 if args.smoke else 20
        trainer.ONLINE_STEPS = 1 if args.smoke else 30
        for f in folds:
            atom(out/'progress.json', dict(stage='pretraining', fold=f, time=time.time()))
            ck = trainer.pretrain(args.model, f, subjects, None, dev)
            test = data.split(f)[2]
            for s in (test[:1] if args.smoke else test):
                check_resource(); atom(out/'progress.json', dict(stage='patient', fold=f, subject=s, time=time.time()))
                trainer.patient(args.model, f, s, ck, subjects, None, dev)
                completed.append(dict(fold=f, subject=s)); gc.collect()
                if dev.type == 'cuda':
                    torch.cuda.empty_cache()
            del ck; gc.collect()
        atom(out/'training_finished.json', dict(status='training_complete_independent_evaluation_required',
                                               completed=completed, smoke=args.smoke, time=time.time()))
    except BaseException as error:
        atom(out/'failure.json', dict(error=str(error), traceback=traceback.format_exc(), automatic_retry=False, time=time.time()))
        raise
    finally:
        lock.close()


if __name__ == '__main__':
    main()
