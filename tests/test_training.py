"""Model and training checks use generated tensors only; no bundled weights/data."""
import importlib.util
import os
from pathlib import Path
import tempfile, unittest
import numpy as np

HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from cardiossm_training.models import BCGnet, CardioSSM, BCG_BP_FSNet
    from cardiossm_training.models.fsnet import normalize_windows
    from cardiossm_training.history import Pool
    from cardiossm_training import data, core
    from cardiossm_training.options import TrainingChoices
    torch.set_num_threads(1)


@unittest.skipUnless(HAS_TORCH, 'optional training dependency torch is not installed')
class ModelTests(unittest.TestCase):
    def test_elapsed_schedule_maps_and_deduplicates_retained_blocks(self):
        d=dict(Tb=np.array([0.,120.,240.,360.,100000.]),k=2)
        np.testing.assert_array_equal(data.schedule('sample',d,4,'A'),[2,4])
        d['reading_indices_4']=np.array([2,3])
        with self.assertRaises(ValueError):data.schedule('sample',d,4,'A')

    def test_bcgnet_prediction_reconstruction_and_backward(self):
        net = BCGnet().train()
        x = torch.randn(3, 1, 1000); f = torch.randn(3, 42)
        p, rec, emb = net(x, f)
        self.assertEqual(tuple(p.shape), (3, 1))
        self.assertEqual(tuple(rec.shape), tuple(x.shape))
        self.assertEqual(tuple(emb.shape), (3, 64))
        (p.square().mean() + .1 * (rec - x).square().mean()).backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None))

    def test_fsnet_forward(self):
        net = BCG_BP_FSNet().eval()
        x = normalize_windows(torch.randn(2, 3, 1000))
        self.assertEqual(tuple(net(x.unsqueeze(2)).shape), (2, 1))

    def test_fsnet_uses_shared_supervised_fit(self):
        from cardiossm_training.current_data import Pool as CurrentPool
        rng = np.random.default_rng(2); n = 8
        d = dict(t=np.arange(n)*20., sig=rng.normal(size=(n,3,1000)).astype(np.float32),
                 fid=np.zeros((n,3,42),np.float32),map=rng.normal(80,3,n).astype(np.float32))
        pool = CurrentPool({'sample':d},['sample'],whole_windows=True)
        tr=pool.ids('sample',np.arange(6));va=pool.ids('sample',np.arange(6,8))
        norm=(np.zeros(42),np.ones(42),pool.Y[tr].mean(),pool.Y[tr].std()+1e-6)
        engine=core.FSNetEngine(pool,norm,torch.device('cpu'));model=BCG_BP_FSNet()
        best,history,steps=engine.fit(model,tr,va,1,1e-4,1,5,False,True)
        self.assertEqual(steps,1);self.assertTrue(np.isfinite(best[0]))

    @unittest.skipUnless(os.environ.get('CARDIOSSM_TEST_CUDA')=='1' and HAS_TORCH and importlib.util.find_spec('mamba_ssm') is not None
                         and torch.cuda.is_available(), 'real Mamba2 execution requires optional Mamba and CUDA')
    def test_cardiossm_native_cuda_forward_backward(self):
        net = CardioSSM().cuda().train()
        x = torch.randn(2, 4, 1, 1000, device='cuda'); f = torch.randn(2, 4, 42, device='cuda')
        p, rec, _ = net(x, f, lengths=[4, 3])
        self.assertEqual(tuple(p.shape), (2, 1))
        (p.square().mean() + .1 * rec.square().mean()).backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None))

    @unittest.skipUnless(os.environ.get('CARDIOSSM_TEST_CUDA')=='1' and HAS_TORCH
                         and importlib.util.find_spec('mamba_ssm') is not None and torch.cuda.is_available(),
                         'native CUDA comparison is opt-in and requires Mamba2/CUDA')
    def test_native_scan_matches_streamed_recurrence(self):
        with torch.no_grad():
            net=CardioSSM().cuda().eval();tokens=torch.randn(2,9,64,device='cuda')
            expected=net.temporal(tokens);states=None;outputs=[]
            for token in tokens.unbind(1):
                value,states=net.temporal_step(token,states);outputs.append(value)
            torch.testing.assert_close(torch.stack(outputs,1),expected,rtol=2e-3,atol=2e-3)

    def test_full_context_never_uses_future_labels_or_other_axis(self):
        t = np.r_[np.arange(20) * 120., 10000 + np.arange(20) * 120.]
        d = dict(t=t, availability_s=t+10, sig=np.zeros((len(t), 3, 1000), np.float32),
                 fid=np.zeros((len(t), 3, 42), np.float32), map=np.arange(len(t), dtype=np.float32))
        d.update(data.run_support(t,t+10,'window_completion'))
        pool = Pool({'sample': d}, ['sample'])
        ids = np.arange(len(pool.Y)); ix, valid = pool.contexts(ids)
        self.assertTrue(np.all(pool.T[ix][valid] <= np.broadcast_to(pool.T[ids, None], ix.shape)[valid]))
        self.assertTrue(np.all((pool.axis[ix] == pool.axis[ids, None])[valid]))
        # The first token after the long gap has no pre-gap context.
        self.assertEqual(int(valid[60].sum()), 1)

    def test_portable_bcgnet_fit_uses_train_val_and_updates(self):
        from cardiossm_training.current_data import Pool as CurrentPool
        rng = np.random.default_rng(1); n = 12
        d = dict(t=np.arange(n)*20., sig=rng.normal(size=(n, 3, 1000)).astype(np.float32),
                 fid=rng.normal(size=(n, 3, 42)).astype(np.float32), map=rng.normal(80, 3, n).astype(np.float32))
        pool = CurrentPool({'sample': d}, ['sample'])
        tr = pool.ids('sample', np.arange(8)); va = pool.ids('sample', np.arange(8, 12))
        norm = (pool.F[tr].mean(0), pool.F[tr].std(0)+1e-6, pool.Y[tr].mean(), pool.Y[tr].std()+1e-6)
        engine = core.BCGnetEngine(pool, norm, torch.device('cpu')); model = BCGnet()
        before = {k: v.clone() for k, v in model.state_dict().items()}
        best, history, steps = engine.fit(model, tr, va, 3, 1e-4, 1, 5, False)
        self.assertEqual(steps, 1); self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(best[0]))
        self.assertTrue(any(not torch.equal(before[k], v) for k, v in model.state_dict().items()))

    def test_reading_label_changes_only_later_predictions(self):
        rng=np.random.default_rng(6);n=180;t=np.arange(n)*20.
        d=dict(t=t,availability_s=t+10,sig=rng.normal(size=(n,3,1000)).astype(np.float32),
               fid=rng.normal(size=(n,3,42)).astype(np.float32),map=rng.normal(80,3,n).astype(np.float32),
               retained_mask=np.ones(n,bool),reading_indices_4=np.array([9]))
        retained=data.retained(d,'sample','windows','window_completion');pool=core.pool_for('bcgnet')({'sample':retained},['sample'])
        tr=pool.ids('sample',retained['itr'])
        norm=(pool.F[tr].mean(0),pool.F[tr].std(0)+1e-6,pool.Y[tr].mean(),pool.Y[tr].std()+1e-6)
        torch.manual_seed(4);ck=dict(state=core.state(BCGnet()),norm=norm,architecture_version=BCGnet.architecture_version,run_gap_basis='window_completion')
        old=(core.ROOT,core.FIT_EPOCHS,core.ONLINE_STEPS,core.NS,core.SEED)
        with tempfile.TemporaryDirectory() as directory:
            try:
                core.FIT_EPOCHS=1;core.ONLINE_STEPS=1;core.NS=[4];core.SEED=0
                predictions=[]
                for case in range(2):
                    altered={k:v.copy() for k,v in d.items()}
                    if case:altered['map'][retained['groups'][9]]-=400
                    core.ROOT=Path(directory)/str(case)
                    (core.ROOT/'bcgnet/f0').mkdir(parents=True)
                    choices=TrainingChoices('windows',1e-4,64,64,'fit',online_sampling='with_replacement',run_gap_basis='window_completion')
                    core.patient('bcgnet',0,'sample',ck,{'sample':altered},None,torch.device('cpu'),choices)
                    with np.load(core.ROOT/'bcgnet/f0/sample.npz') as z:predictions.append(z['IP4'].copy())
                np.testing.assert_array_equal(predictions[0][:10],predictions[1][:10])
                self.assertGreater(float(np.max(abs(predictions[0][10:]-predictions[1][10:]))),0.)
            finally:
                core.ROOT,core.FIT_EPOCHS,core.ONLINE_STEPS,core.NS,core.SEED=old


if __name__ == '__main__':
    unittest.main()
