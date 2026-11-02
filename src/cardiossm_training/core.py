"""Shared three-model training, initial fit and periodic online replay."""
from pathlib import Path
import os,time,json,hashlib
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,torch
from torch import nn
from . import data
from .history import Pool as HistoryPool
from .current_data import Pool as CurrentPool
from .models import build
from .models.fsnet import normalize_windows
from functools import partial
from .streaming import StreamPredictor
from .options import TrainingChoices
ROOT=None;SEED=0;PRETRAIN_EPOCHS=3;FIT_EPOCHS=20;ONLINE_STEPS=30
NS=[4];BS=256
F_loss=nn.HuberLoss(delta=1.)
def state(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}

class Engine:
    def __init__(self,pool,norm,dev):
        self.pool=pool;self.norm=norm;self.dev=dev
        self.features=torch.as_tensor(data.nz(pool.F,norm),device=dev)
        self.cache=None;self.refresh_count=0;self.cache_present=torch.zeros(len(pool.Y),device=dev,dtype=torch.bool)
    @torch.no_grad()
    def refresh(self,net,ids=None):
        net.eval();n=len(self.pool.Y)
        if self.cache is None:self.cache=torch.zeros(n,64,device=self.dev,dtype=torch.bfloat16)
        if ids is None:needed=np.arange(n)
        else:
            # Scope only; all queried history embeddings still use current weights.
            used=np.zeros(n,bool)
            for b in range(0,len(ids),8192):
                ix,valid=self.pool.contexts(np.asarray(ids)[b:b+8192]);used[ix[valid]]=True
            needed=np.flatnonzero(used)
        with torch.autocast(self.dev.type,dtype=torch.bfloat16,enabled=self.dev.type=='cuda'):
            for b in range(0,len(needed),512):
                q=needed[b:b+512];x=torch.as_tensor(self.pool.X[q].astype(np.float32),device=self.dev)
                emb,_=net.wave(x,reconstruct=False);self.cache[q]=emb.to(self.cache.dtype);self.cache_present[q]=True
        self.refresh_count+=1
    def forward(self,net,ids,training=False,ignore=False,reconstruct=False):
        ids=np.asarray(ids,dtype=np.int64);ix,valid=self.pool.contexts(ids,ignore);index=torch.as_tensor(ix,device=self.dev);mask=torch.as_tensor(valid,device=self.dev);length=mask.sum(1)
        assert bool(self.cache_present[index][mask].all())
        h=self.cache[index];rec=None;x=None
        if training:
            x=torch.as_tensor(self.pool.X[ids].astype(np.float32),device=self.dev)
            live,rec=net.wave(x,reconstruct=reconstruct)
            # Current and all sampled past targets share the same trainable CNN.
            # History outside this optimizer block is a truncated-gradient cache.
            uid,first=np.unique(ids,return_index=True);uid=torch.as_tensor(uid,device=self.dev);first=torch.as_tensor(first,device=self.dev)
            pos=torch.searchsorted(uid,index);clamped=pos.clamp_max(len(uid)-1);hit=(pos<len(uid))&(uid[clamped]==index)
            h=torch.where(hit[...,None],live[first][clamped],h)
        tokens=net.token_projection(torch.cat([h,self.features[index]],-1));tokens=tokens*mask[...,None]
        hidden=net.temporal(tokens);p=net.head(hidden[torch.arange(len(ids),device=self.dev),length-1])
        return p,rec,x
    @torch.no_grad()
    def predict(self,net,ids,refresh=True,ignore=False):
        if refresh:self.refresh(net,ids)
        net.eval();out=[]
        with torch.autocast(self.dev.type,dtype=torch.bfloat16,enabled=self.dev.type=='cuda'):
            for b in range(0,len(ids),512):out.append(self.forward(net,ids[b:b+512],ignore=ignore)[0].float().cpu().numpy()[:,0])
        return np.concatenate(out)*self.norm[3]+self.norm[2]
    def fit(self,net,tr,va,nc,lr,EP,pat,onecycle,eval_first=False):
        bs=getattr(self,'batch_size',BS)
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4);steps=(len(tr)+bs-1)//bs
        sch=torch.optim.lr_scheduler.OneCycleLR(opt,lr,total_steps=EP*steps) if onecycle else None
        def val():return float(np.mean(abs(self.predict(net,va).reshape(-1,nc).mean(1)-self.pool.Y[va].reshape(-1,nc)[:,0])))
        fit_start=time.time()
        def progress(**values):
            if hasattr(self,'audit_path'):
                tmp=self.audit_path.with_suffix('.tmp');tmp.write_text(json.dumps(dict(elapsed_s=time.time()-fit_start,train_samples=len(tr),validation_samples=len(va),steps_per_epoch=steps,max_epochs=EP,context_audit=self.pool.context_audit,**values)));tmp.replace(self.audit_path)
        best=(val(),state(net),-1) if eval_first else (float('inf'),None,-1);history=[];bad=0;step_count=0
        for ep in range(EP):
            self.refresh(net,tr)
            net.train();seen=[];nsteps=0
            pm=torch.randperm(len(tr)).numpy()
            for b in range(0,len(tr),bs):
                data.CHECK_RESOURCE()
                ids=tr[pm[b:b+bs]]
                yy=torch.as_tensor(((self.pool.Y[ids]-self.norm[2])/self.norm[3])[:,None],device=self.dev)
                with torch.autocast(self.dev.type,dtype=torch.bfloat16,enabled=self.dev.type=='cuda'):
                    p,rec,x=self.forward(net,ids,True,reconstruct=True);loss=F_loss(p.float(),yy)
                    if rec is not None:loss=loss+.1*nn.functional.mse_loss(rec.float(),x)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fit loss')
                opt.zero_grad();loss.backward();opt.step()
                if sch:sch.step()
                seen.append(ids);step_count+=1;nsteps+=1
                if nsteps%25==0:progress(epoch=ep,epoch_step=nsteps,optimizer_steps=step_count,complete=False)
                if nsteps%100==0:print('TRAIN_STEP',ep,nsteps,'of',steps,flush=True)
            assert nsteps==steps and np.array_equal(np.sort(np.concatenate(seen)),np.sort(tr))
            vl=val();
            if not np.isfinite(vl):raise FloatingPointError('Nonfinite validation MAE')
            history.append(vl);progress(epoch=ep,epoch_step=nsteps,optimizer_steps=step_count,val_mae=vl,history=history,complete=False);print('EPOCH',ep,round(vl,3),'steps',nsteps,flush=True)
            if vl<best[0]:best=(vl,state(net),ep);bad=0
            else:
                bad+=1
                if bad>=pat:break
        net.load_state_dict(best[1]);progress(epoch=ep,epoch_step=nsteps,optimizer_steps=step_count,best_epoch=best[2],history=history,complete=True)
        return best,history,step_count

def run_path(arch, fold):
    if ROOT is None:
        raise ValueError('an explicit external output directory is required')
    root=Path(ROOT).resolve()
    if root.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('training outputs must stay outside the code repository')
    return root/arch/f'f{fold}'


def pretrain(arch,f,D,unused,dev,choices=None):
    choices=choices or TrainingChoices()
    if arch=='fsnet' and not choices.fsnet_batch_size:
        raise ValueError('FSNet batch size must be explicitly specified in whole windows')
    path=run_path(arch,f);path.mkdir(parents=True,exist_ok=True)
    if (path/'pretrained.pt').exists():raise FileExistsError('Checkpoint already exists; choose a fresh run directory')
    trsub,vasub,tesub=data.split(f);prepared=data.pretraining_data(D,trsub+vasub,choices.run_gap_basis);pool=pool_for(arch)(prepared,trsub+vasub)
    tr=np.concatenate([pool.ids(s,np.arange(0,pool.count[s],4)) for s in trsub]);va=np.concatenate([pool.ids(s,np.arange(0,pool.count[s],20)) for s in vasub])
    norm=(np.nanmean(pool.F[tr],0),np.nanstd(pool.F[tr],0)+1e-6,pool.Y[tr].mean(),pool.Y[tr].std()+1e-6)
    torch.manual_seed(SEED);net=build(arch).to(dev);engine=engine_for(arch)(pool,norm,dev);engine.batch_size=choices.fsnet_batch_size if arch=='fsnet' else BS;engine.audit_path=path/'pretrain_progress.json';t=time.time();best,hist,steps=engine.fit(net,tr,va,pool.axes,1e-3,PRETRAIN_EPOCHS,99,True)
    ck=dict(state=best[1],norm=norm,arch=arch,architecture_version=net.architecture_version,fold=f,seed=SEED,train=trsub,val=vasub,test=tesub,pretrain_samples=len(tr),validation_samples=len(va),best_epoch=best[2],val_mae=best[0],history=hist,optimizer_steps=steps,parameters=sum(p.numel() for p in net.parameters()),elapsed_s=time.time()-t,context_audit=pool.context_audit,training_axis_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),validation_axis_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest(),history_report=pool.report,history_policy=('all_causal_within_run' if arch=='cardiossm' else 'current_window_only'),initialization=f'fresh random seed{SEED}; no prior model weights loaded',gradient_strategy=('live sampled CNN targets; other context embeddings detached and refreshed; full projection/SSM gradients' if arch=='cardiossm' else 'independently shuffled current axis-window targets, full network gradients'))
    ck.update(batch_size=engine.batch_size,sample_unit='three_axis_window' if arch=='fsnet' else 'axis_window',run_gap_basis=choices.run_gap_basis)
    torch.save(ck,path/'pretrained.pt');(path/'pretrain.json').write_text(json.dumps({k:v for k,v in ck.items() if k not in ['state','norm']},indent=2));return ck

def adapted_predictions(engine, net, d, pool, subject, replay, hours, setting, choices,
                        seed=0, steps=30):
    """Predict once in time order; optimizer/replay never mutate inference H."""
    choices.validate('cardiossm' if isinstance(pool,HistoryPool) else 'bcgnet',
                     ['initial_periodic' if setting=='A' else 'periodic_only'])
    pos=data.schedule(subject,d,hours,setting)
    readings=np.full(len(d['Tb']),np.nan)
    readings[pos]=d['Yb'][pos]+data.noise(subject,hours,0 if setting=='A' else 5,len(pos))
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    net.prediction_only=True
    opt=torch.optim.AdamW(net.parameters(),lr=choices.online_learning_rate,weight_decay=1e-4)
    stream=StreamPredictor(net,pool,engine.norm,engine.dev) if isinstance(pool,HistoryPool) else None
    raw=np.full(len(d['t']),np.nan)
    def predict(windows):
        ids=pool.ids(subject,np.asarray(windows,dtype=int))
        result=stream.predict(ids) if stream is not None else engine.predict(net,ids)
        raw[windows]=result.reshape(-1,pool.axes).mean(1)
    prefix=np.arange(int(d['groups'][int(pos[0])][0]))
    if len(prefix):predict(prefix)
    memory=[];labels=[];updates=[];digest=hashlib.sha256()
    for read,end in zip(pos,list(pos[1:])+[len(d['Tb'])]):
        wi=d['groups'][int(read)]
        predict(wi)  # Entire reading block, with old weights and carried H.
        ids=pool.ids(subject,wi)
        memory.extend(ids.tolist());labels.extend([float(readings[read])]*len(ids))
        mi=np.asarray(memory);my=np.asarray(labels)
        needed=np.unique(np.r_[mi,replay]) if setting=='A' else np.unique(mi)
        engine.refresh(net,needed);net.train()
        for _ in range(steps):
            data.CHECK_RESOURCE()
            replace=choices.online_sampling=='with_replacement'
            if not replace and (choices.memory_samples>len(mi) or (setting=='A' and choices.replay_samples>len(replay))):
                raise ValueError('without-replacement sample counts exceed currently available support')
            selection=rng.choice(len(mi),size=choices.memory_samples,replace=replace)
            batch=mi[selection];targets=my[selection]
            if setting=='A':
                rr=rng.choice(len(replay),size=choices.replay_samples,replace=replace)
                batch=np.r_[batch,replay[rr]];targets=np.r_[targets,pool.Y[replay[rr]]]
            digest.update(batch.tobytes());digest.update(targets.tobytes())
            yy=torch.as_tensor(((targets-engine.norm[2])/engine.norm[3]).astype(np.float32),device=engine.dev)[:,None]
            with torch.autocast(engine.dev.type,dtype=torch.bfloat16,enabled=engine.dev.type=='cuda'):
                p,_,_=engine.forward(net,batch,True,reconstruct=False);loss=F_loss(p.float(),yy)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite online loss')
            opt.zero_grad();loss.backward();opt.step()
        # The stream object and both kinds of state survive all training steps.
        if int(read)+1<int(end):predict(np.concatenate(d['groups'][int(read)+1:int(end)]))
        updates.append(dict(block=int(read),memory_samples=len(mi),steps=steps))
    if not np.isfinite(raw).all():raise AssertionError('unpredicted windows in online run')
    return raw,pos,readings,dict(sampling_sha256=digest.hexdigest(),updates=updates,
        noise_sd=0 if setting=='A' else 5,inference_state='carried_across_readings' if stream else 'windowwise')


def patient(arch,f,s,ck,D,unused,dev,choices=None,conditions=('initial_periodic',)):
    choices=choices or TrainingChoices();choices.validate(arch,conditions)
    path=run_path(arch,f);path.mkdir(parents=True,exist_ok=True);done=path/f'{s}.json'
    if done.exists():raise FileExistsError('Patient output already exists; no automatic resume')
    t0=time.time();d=data.retained(D[s],s,choices.initial_split_unit,choices.run_gap_basis)
    pool=pool_for(arch)({s:d},[s]);norm=ck['norm'];net=build(arch).to(dev)
    if ck.get('architecture_version') != net.architecture_version:
        raise ValueError('starting checkpoint architecture/history policy differs; fresh matching pretraining is required')
    if ck.get('run_gap_basis') != choices.run_gap_basis:
        raise ValueError('starting checkpoint must declare the same run_gap_basis')
    engine=engine_for(arch)(pool,norm,dev);engine.batch_size=choices.fsnet_batch_size if arch=='fsnet' else BS
    engine.audit_path=path/f'{s}_ft_progress.json'
    allids=np.arange(len(pool.Y));tr=pool.ids(s,d['itr']);va=pool.ids(s,d['iva'])
    out={k:d[k] for k in ('Tb','Yb','te')}
    out.update(run_id=d['rid'],window_run_id=d['window_run_id'],run_gap_basis=np.asarray(d['run_gap_basis']))
    out.update(raw_t=d['t'],raw_bid=d['bid'],ub=d['ub'],availability_s=d['availability_s'],raw_reference=d['y'])
    if 'HRb' in d:out['HRb']=d['HRb']
    meta=dict(subject=s,arch=arch,architecture_version=net.architecture_version,fold=f,seed=SEED,
        context_audit=pool.context_audit,history_report=pool.report[s],schedules={},
        choices=vars(choices),conditions=list(conditions),no_bp_token_inputs=True)
    ftstate=None
    if any(c in ('initial_only','initial_periodic') for c in conditions):
        torch.manual_seed(SEED);net.load_state_dict(ck['state']);net.prediction_only=False
        best,hist,steps=engine.fit(net,tr,va,pool.axes,1e-4,FIT_EPOCHS,5,False,True)
        ftstate=state(net)
        meta.update(ft_epoch=best[2],ft_history=hist,ft_optimizer_steps=steps)
    for condition in conditions:
        initial=condition in ('initial_only','initial_periodic')
        periodic=condition in ('periodic_only','initial_periodic')
        for hours in (NS if periodic else [None]):
            torch.manual_seed(SEED);net.load_state_dict(ftstate if initial else ck['state'])
            if periodic:
                replay=tr if choices.replay_source=='fit' else pool.ids(s,d['calibration_windows'])
                raw,pos,readings,info=adapted_predictions(engine,net,d,pool,s,replay if initial else np.array([],int),
                    hours,'A' if initial else 'B',choices,SEED,ONLINE_STEPS)
                tag=('IP' if initial else 'P5_')+str(hours)
                out[tag+'_readings']=readings
            else:
                # Fresh state for each standalone fixed-weight condition.
                raw=(StreamPredictor(net,pool,norm,dev).predict(allids) if arch=='cardiossm'
                     else engine.predict(net,allids)).reshape(-1,pool.axes).mean(1)
                pos=np.array([],dtype=int);info={};tag='I' if initial else 'PT'
            block=np.array([np.mean(raw[g]) for g in d['groups']])
            out[tag+'_raw']=raw;out[tag+'_block']=block;out[tag]=data.smooth(block,d['Tb'],d['rid'])
            out[tag+'_positions']=pos
            out[tag+'_test_mask']=d['te'] if initial or choices.no_initial_test_scope=='final70' else np.ones(len(d['Tb']),bool)
            out[tag+'_calibration_mask']=~d['te'] if initial else np.zeros(len(d['Tb']),bool)
            meta['schedules'][tag]=dict(condition=condition,interval_h=hours,**info)
    np.savez_compressed(path/f'{s}.npz',**out)
    meta['elapsed_s']=time.time()-t0;done.write_text(json.dumps(meta,indent=2))


class BCGnetEngine(Engine):
    @torch.no_grad()
    def refresh(self,net,ids=None):
        net.eval();self.refresh_count+=1

    def forward(self,net,ids,training=False,ignore=False,reconstruct=False):
        ids=np.asarray(ids,dtype=np.int64)
        x=torch.as_tensor(self.pool.X[ids],device=self.dev)
        net.prediction_only=not reconstruct
        prediction,rec,_=net(x,self.features[ids])
        return prediction,rec,x


class FSNetEngine(BCGnetEngine):
    def forward(self,net,ids,training=False,ignore=False,reconstruct=False):
        x=torch.as_tensor(self.pool.X[np.asarray(ids,dtype=np.int64)],device=self.dev)
        p=net(normalize_windows(x).unsqueeze(2))
        return p,None,x


def engine_for(arch):return {'cardiossm':Engine,'bcgnet':BCGnetEngine,'fsnet':FSNetEngine}[arch]

def pool_for(arch):return HistoryPool if arch=='cardiossm' else partial(CurrentPool,whole_windows=arch=='fsnet')
