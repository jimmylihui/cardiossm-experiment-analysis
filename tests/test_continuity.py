"""Generated CPU parameter fixtures test production recurrence and runner paths.

These tests do not substitute a mock CUDA kernel or claim native Mamba coverage.
The small encoder/parameter fixtures isolate state transport and indexing.
"""
import copy
import importlib.util
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from cardiossm_training.models.cardiossm import CardioSSM
    from cardiossm_training.streaming import StreamPredictor, mamba2_step
    from cardiossm_training.history import Pool
    from cardiossm_training import core, data
    from cardiossm_training.options import TrainingChoices
    torch.set_num_threads(1)

    class GatedNorm(nn.Module):
        def __init__(self):
            super().__init__();self.weight=nn.Parameter(torch.ones(128))
        def forward(self,y,g):
            y=y*F.silu(g)
            return y*torch.rsqrt(y.square().mean(-1,keepdim=True)+1e-5)*self.weight

    class ParameterFixture(nn.Module):
        def __init__(self):
            super().__init__()
            self.ngroups=1;self.d_ssm=self.d_inner=128;self.d_state=32
            self.D_has_hdim=False;self.nheads=4;self.headdim=32;self.d_conv=4
            self.in_proj=nn.Linear(64,324,bias=False)
            self.conv1d=nn.Conv1d(192,192,4,groups=192,padding=3)
            self.dt_bias=nn.Parameter(torch.full((4,),-3.))
            self.A_log=nn.Parameter(torch.zeros(4));self.D=nn.Parameter(torch.ones(4))
            self.norm=GatedNorm();self.out_proj=nn.Linear(128,64,bias=False)

    class EncoderFixture(nn.Module):
        def __init__(self):
            super().__init__();self.scale=nn.Parameter(torch.linspace(.5,1.5,64))
        def forward(self,x,reconstruct=False):
            return x.mean(-1)*self.scale, x*self.scale.mean() if reconstruct else None

    class SignalModelFixture(CardioSSM):
        def __init__(self):
            nn.Module.__init__(self)
            self.wave=EncoderFixture();self.token_projection=nn.Linear(106,64)
            self.norms=nn.ModuleList([nn.LayerNorm(64) for _ in range(2)])
            self.ssm=nn.ModuleList([ParameterFixture() for _ in range(2)])
            self.final_norm=nn.LayerNorm(64);self.head=nn.Linear(64,1)
            self.dropout=nn.Dropout(0.);self.prediction_only=False
        def temporal(self,tokens):
            # Differentiable sequence path for generated replay tests only.
            z=tokens
            for norm,block in zip(self.norms,self.ssm):
                outputs=[];states=None
                for token in norm(z).unbind(1):
                    out,states=mamba2_step(block,token,states);outputs.append(out)
                z=z+torch.stack(outputs,1)
            return self.final_norm(z)

    def oracle(block,tokens):
        """Independent full-sequence convolution + direct state equation."""
        v=block.in_proj(tokens)
        gate,xbc,dt=v.split([128,192,4],-1)
        filtered=F.silu(F.conv1d(xbc.transpose(1,2),block.conv1d.weight,block.conv1d.bias,
                                padding=3,groups=192)[...,:tokens.shape[1]]).transpose(1,2)
        signal,b,c=filtered.split([128,32,32],-1)
        h=torch.zeros(len(tokens),4,32,32);outputs=[]
        for i in range(tokens.shape[1]):
            delta=F.softplus(dt[:,i]+block.dt_bias)
            x=signal[:,i].reshape(-1,4,32)
            injection=torch.einsum('bh,bhp,bn->bhpn',delta,x,b[:,i])
            h=torch.einsum('bh,bhpn->bhpn',torch.exp(-block.A_log.exp()*delta),h)+injection
            y=torch.einsum('bhpn,bn->bhp',h,c[:,i])+block.D[None,:,None]*x
            outputs.append(block.out_proj(block.norm(y.flatten(1),gate[:,i])))
        return torch.stack(outputs,1),h


@unittest.skipUnless(HAS_TORCH,'optional torch dependency unavailable')
class ContinuityTests(unittest.TestCase):
    def recording(self,t):
        rng=np.random.default_rng(4);n=len(t)
        support=data.run_support(np.asarray(t),np.asarray(t)+10,'window_completion')
        return dict(t=np.asarray(t,float),availability_s=np.asarray(t,float)+10,
            sig=rng.normal(size=(n,3,1000)).astype(np.float32),
            fid=rng.normal(size=(n,3,42)).astype(np.float32),map=np.full(n,80,np.float32),
            retained_mask=np.ones(n,bool),recording_start_s=np.asarray(float(t[0])),
            window_run_id=support['window_run_id'],run_gap_basis='window_completion')

    def test_every_10s_window_present_and_no_phase_reset(self):
        for t in [np.arange(24)*10.,np.r_[np.arange(12)*10.,
                  np.concatenate([b*120+np.arange(6)*10 for b in range(1,17)]),17*120+np.arange(12)*10.]]:
            p=Pool({'sample':self.recording(t)},['sample'])
            for axis in range(3):
                q=p.ids('sample',[len(t)-1])[axis];ix,valid=p.contexts([q])
                np.testing.assert_array_equal(ix[0,valid[0]],np.arange(len(t))*3+axis)
            self.assertTrue(p.context_audit['complete_history'])

    def test_only_true_gap_resets_and_patients_axes_are_separate(self):
        t=np.array([0.,10.,1810.,3621.])
        p=Pool({'a':self.recording(t),'b':self.recording(t)},['a','b'])
        for subject in ('a','b'):
            for i,count in enumerate([1,2,3,1]):
                ix,valid=p.contexts(p.ids(subject,[i]))
                self.assertEqual(valid.sum(1).tolist(),[count]*3)
                for a in range(3):self.assertTrue(np.all(p.axis[ix[a,valid[a]]]==a))

    def test_production_step_matches_independent_equations(self):
        torch.manual_seed(7);b=ParameterFixture().eval();tokens=torch.randn(2,13,64)
        expected,h=oracle(b,tokens)
        actual=[];states=None
        for token in tokens.unbind(1):
            output,states=mamba2_step(b,token,states);actual.append(output)
        torch.testing.assert_close(torch.stack(actual,1),expected,rtol=1e-5,atol=1e-6)
        torch.testing.assert_close(states[1],h,rtol=1e-5,atol=1e-6)

    def test_stream_chunking_weight_update_and_replay_isolation(self):
        torch.manual_seed(8);net=SignalModelFixture().eval()
        p=Pool({'sample':self.recording(np.arange(24)*10.)},['sample'])
        norm=(np.zeros(42),np.ones(42),0.,1.);ids=np.arange(len(p.Y));dev=torch.device('cpu')
        whole=StreamPredictor(net,p,norm,dev).predict(ids)
        stream=StreamPredictor(net,p,norm,dev)
        prefix=stream.predict(ids[:36]);saved=copy.deepcopy(stream.streams)
        with torch.no_grad():net.ssm[0].in_proj.weight.mul_(1.3)
        engine=core.Engine(p,norm,dev);engine.refresh(net,ids[:30]);net.train()
        loss=engine.forward(net,ids[:6],training=True)[0].square().mean();loss.backward()
        for key in saved:
            for old,new in zip(saved[key][2],stream.streams[key][2]):
                for x,y in zip(old,new):torch.testing.assert_close(x,y,rtol=0,atol=0)
        suffix=stream.predict(ids[36:])
        np.testing.assert_allclose(prefix,whole[:36],rtol=1e-5,atol=1e-6)
        replayed=StreamPredictor(net,p,norm,dev).predict(ids)
        self.assertGreater(np.max(abs(suffix-replayed[36:])),1e-6)
        fixed=SignalModelFixture().eval();fixed.load_state_dict(net.state_dict())
        a=StreamPredictor(fixed,p,norm,dev);one=a.predict(ids)
        b=StreamPredictor(fixed,p,norm,dev)
        np.testing.assert_allclose(np.r_[b.predict(ids[:18]),b.predict(ids[18:45]),b.predict(ids[45:])],one,rtol=1e-5,atol=1e-6)
        with self.assertRaises(ValueError):b.predict(ids[-3:])
        with self.assertRaises(ValueError):StreamPredictor(net,p,norm,dev).predict(ids[3:6])

    def test_inference_state_resets_at_true_gap(self):
        torch.manual_seed(3);net=SignalModelFixture().eval();dev=torch.device('cpu')
        recording=self.recording(np.array([0.,10.,20.,4000.,4010.]))
        p=Pool({'sample':recording},['sample']);norm=(np.zeros(42),np.ones(42),0.,1.)
        prediction=StreamPredictor(net,p,norm,dev).predict(np.arange(len(p.Y)))
        suffix={k:(v[3:] if isinstance(v,np.ndarray) and v.ndim else v) for k,v in recording.items()}
        q=Pool({'sample':suffix},['sample'])
        reset=StreamPredictor(net,q,norm,dev).predict(np.arange(len(q.Y)))
        np.testing.assert_allclose(prediction[9:],reset,rtol=1e-5,atol=1e-6)

    def test_reading_runner_changes_only_later_predictions_with_carried_state(self):
        d=data.retained(self.recording(np.arange(72)*10.),'sample','windows','window_completion')
        choices=TrainingChoices('windows',1e-3,2,2,'fit',online_sampling='with_replacement',run_gap_basis='window_completion')
        torch.manual_seed(5);initial=SignalModelFixture().state_dict();outputs=[]
        for change in (0.,50.):
            altered=copy.deepcopy(d);altered['Yb'][1]+=change
            p=Pool({'sample':altered},['sample']);norm=(np.zeros(42),np.ones(42),80.,3.)
            net=SignalModelFixture();net.load_state_dict(initial);engine=core.Engine(p,norm,torch.device('cpu'))
            replay=p.ids('sample',np.arange(6))
            with patch.object(data,'schedule',return_value=np.array([1,3])):
                raw,_,_,audit=core.adapted_predictions(engine,net,altered,p,'sample',replay,4,'A',choices,steps=1)
            self.assertEqual(audit['inference_state'],'carried_across_readings');outputs.append(raw)
        np.testing.assert_array_equal(outputs[0][:24],outputs[1][:24])
        self.assertGreater(np.max(abs(outputs[0][24:]-outputs[1][24:])),0.)

    def test_split_is_explicit_and_pretraining_excludes_masked_labels(self):
        d=self.recording(np.concatenate([b*120+np.arange(12 if b==1 else 6)*10 for b in range(10)]))
        a=data.retained(d,'sample','blocks','window_completion');b=data.retained(d,'sample','windows','window_completion')
        self.assertFalse(set(a['bid'][a['itr']])&set(a['bid'][a['iva']]))
        self.assertTrue(set(b['bid'][b['itr']])&set(b['bid'][b['iva']]))
        with self.assertRaises(ValueError):TrainingChoices().validate('cardiossm',['initial_periodic'])
        d['map'][0]=999.;d['retained_mask'][0]=False
        self.assertNotIn(999.,data.pretraining_data({'sample':d},['sample'],'window_completion')['sample']['map'])

    def test_periodic_only_never_replays_or_reads_unscheduled_labels(self):
        self.assertTrue(np.array_equal(data.noise('sample',4,5,8),data.noise('sample',4,5,8)))
        self.assertFalse(np.array_equal(data.noise('sample',4,5,8),data.noise('sample',8,5,8)))
        d=data.retained(self.recording(np.arange(72)*10.),'sample',run_gap_basis='window_completion')
        np.testing.assert_array_equal(data.schedule('sample',d,4,'B'),[0])
        schedule=dict(Tb=np.array([120.,14400.,14520.,28800.]),k=1,recording_start_s=0.)
        np.testing.assert_array_equal(data.schedule('sample',schedule,4,'B'),[0,1,3])
        choices=TrainingChoices(online_learning_rate=1e-3,memory_samples=2,no_initial_test_scope='all',online_sampling='with_replacement',run_gap_basis='window_completion')
        torch.manual_seed(6);initial=SignalModelFixture().state_dict();outputs=[]
        for change in (0.,100.):
            altered=copy.deepcopy(d);altered['y'][12:]+=change;altered['Yb'][1:]+=change
            p=Pool({'sample':altered},['sample']);norm=(np.zeros(42),np.ones(42),80.,3.)
            net=SignalModelFixture();net.load_state_dict(initial);engine=core.Engine(p,norm,torch.device('cpu'))
            raw,_,_,_=core.adapted_predictions(engine,net,altered,p,'sample',np.array([],int),4,'B',choices,steps=1)
            outputs.append(raw)
        np.testing.assert_array_equal(*outputs)

    def test_periodic_only_patient_skips_initial_fit(self):
        recording=self.recording(np.arange(72)*10.)
        net=SignalModelFixture();ck=dict(state=core.state(net),norm=(np.zeros(42),np.ones(42),80.,3.),architecture_version=net.architecture_version,run_gap_basis='window_completion')
        choices=TrainingChoices(online_learning_rate=1e-3,memory_samples=2,
            no_initial_test_scope='all',online_sampling='with_replacement',run_gap_basis='window_completion')
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(core,'ROOT',Path(directory)),patch.object(core,'ONLINE_STEPS',1),patch.object(core,'NS',[4]),\
                 patch.object(core,'build',return_value=net),patch.object(core.Engine,'fit',side_effect=AssertionError('initial fit forbidden')):
                core.patient('cardiossm',0,'sample',ck,{'sample':recording},None,torch.device('cpu'),choices,['periodic_only'])
            with np.load(Path(directory)/'cardiossm/f0/sample.npz') as z:
                self.assertTrue(z['P5_4_test_mask'].all())
                self.assertFalse(z['P5_4_calibration_mask'].any())
                np.testing.assert_array_equal(z['P5_4_positions'],[0])


if __name__=='__main__':unittest.main()
