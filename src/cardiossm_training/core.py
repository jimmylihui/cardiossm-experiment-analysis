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
ROOT=Path('outputs/training');SEED=0;PRETRAIN_EPOCHS=3;FIT_EPOCHS=20;ONLINE_STEPS=30
NS=[4,8,12,24];BS=256
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
                emb,_=net.wave(x,reconstruct=False);self.cache[q]=emb;self.cache_present[q]=True
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
        opt=torch.optim.AdamW(net.parameters(),lr=lr,weight_decay=1e-4);steps=(len(tr)+BS-1)//BS
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
            for b in range(0,len(tr),BS):
                data.CHECK_RESOURCE()
                ids=tr[pm[b:b+BS]]
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

def pretrain(arch,f,D,unused,dev):
    path=ROOT/arch/f'f{f}';path.mkdir(parents=True,exist_ok=True)
    if (path/'pretrained.pt').exists():raise FileExistsError('Checkpoint already exists; choose a fresh run directory')
    trsub,vasub,tesub=data.split(f);pool=pool_for(arch)(D,trsub+vasub)
    tr=np.concatenate([pool.ids(s,np.arange(0,pool.count[s],4)) for s in trsub]);va=np.concatenate([pool.ids(s,np.arange(0,pool.count[s],20)) for s in vasub])
    norm=(np.nanmean(pool.F[tr],0),np.nanstd(pool.F[tr],0)+1e-6,pool.Y[tr].mean(),pool.Y[tr].std()+1e-6)
    torch.manual_seed(SEED);net=build(arch).to(dev);engine=engine_for(arch)(pool,norm,dev);engine.audit_path=path/'pretrain_progress.json';t=time.time();best,hist,steps=engine.fit(net,tr,va,pool.axes,1e-3,PRETRAIN_EPOCHS,99,True)
    ck=dict(state=best[1],norm=norm,arch=arch,architecture_version=net.architecture_version,fold=f,seed=SEED,train=trsub,val=vasub,test=tesub,pretrain_samples=len(tr),validation_samples=len(va),best_epoch=best[2],val_mae=best[0],history=hist,optimizer_steps=steps,parameters=sum(p.numel() for p in net.parameters()),elapsed_s=time.time()-t,context_audit=pool.context_audit,training_axis_indices_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),validation_axis_indices_sha256=hashlib.sha256(va.tobytes()).hexdigest(),history_report=pool.report,history_policy=('all_causal_within_run' if arch=='cardiossm' else 'current_window_only'),initialization=f'fresh random seed{SEED}; no prior model weights loaded',gradient_strategy=('live sampled CNN targets; other context embeddings detached and refreshed; full projection/SSM gradients' if arch=='cardiossm' else 'independently shuffled current axis-window targets, full network gradients'))
    torch.save(ck,path/'pretrained.pt');(path/'pretrain.json').write_text(json.dumps({k:v for k,v in ck.items() if k not in ['state','norm']},indent=2));return ck

def patient(arch,f,s,ck,D,unused,dev):
    path=ROOT/arch/f'f{f}';done=path/f'{s}.json'
    if done.exists():raise FileExistsError('Patient output already exists; no automatic resume')
    t0=time.time();d=data.retained(D[s],s);pool=pool_for(arch)({s:d},[s]);norm=ck['norm'];net=build(arch).to(dev);engine=engine_for(arch)(pool,norm,dev);engine.audit_path=path/f'{s}_ft_progress.json';net.load_state_dict(ck['state']);allids=np.arange(len(pool.Y));Y=pool.Y
    out={k:d[k] for k in ['Tb','Yb','te']};out.update(availability_s=d['availability_s'],yma=data.smooth(d['Yb'],d['Tb'],d['rid']),raw_t=d['t'],raw_bid=d['bid'],ub=d['ub'])
    def flatids(ids):return pool.ids(s,np.asarray(ids))
    def agg(p):return data.smooth(np.array([np.mean(p[g]) for g in d['groups']]),d['Tb'],d['rid'])
    tr=flatids(d['itr']);va=flatids(d['iva']);torch.manual_seed(SEED);np.random.seed(SEED);net.load_state_dict(ck['state']);net.prediction_only=False
    best,hist,steps=engine.fit(net,tr,va,pool.axes,1e-4,FIT_EPOCHS,5,False,True);ftstate=state(net)
    torch.save(dict(state=ftstate,norm=norm,arch=arch,fold=f,seed=SEED,ft_epoch=best[2],history_policy=('all_causal_within_run' if arch=='cardiossm' else 'current_window_only'),architecture_version=net.architecture_version),path/f'{s}_ft.pt')
    pI=engine.predict(net,allids).reshape(-1,pool.axes).mean(1);out['I']=agg(pI)
    meta=dict(context_audit=pool.context_audit,history_policy=('all_causal_within_run' if arch=='cardiossm' else 'current_window_only'),all_training_stages_full_history=(arch=='cardiossm'),subject=s,arch=arch,architecture_version=net.architecture_version,fold=f,seed=SEED,history_source=('same patient/axis/phase full history after >30min gap reset' if arch=='cardiossm' else 'current window'),no_bp_token_inputs=True,history_report=pool.report[s],ft_epoch=best[2],ft_validation_mae=best[0],ft_history=hist,ft_optimizer_steps=steps,calibration_train_axis_indices=tr.tolist(),calibration_val_axis_indices=va.tolist(),schedules={},gradient_strategy=('live sampled CNN targets; detached older embeddings refreshed before/after reading updates; full projection/SSM gradients' if arch=='cardiossm' else 'current axis-window targets, full network gradients'))
    for setting,sd in [('A',0)]:
        for N in NS:
            tag=('IP' if setting=='A' else f'P{sd}_')+str(N);pos=data.schedule(s,d,N,setting);readings=d['Yb']+data.noise(s,N,sd,len(d['Tb']))
            torch.manual_seed(SEED);np.random.seed(SEED);net.load_state_dict(ftstate if setting=='A' else ck['state']);net.prediction_only=True
            opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=1e-4);raw=pI.copy();mem=[];labels=[];bounds=list(pos[1:])+[len(d['Tb'])];updates=[];sampling_digest=hashlib.sha256();losses=[]
            for read,end in zip(pos,bounds):
                wi=d['groups'][int(read)];ii=flatids(wi);raw[wi]=engine.predict(net,ii).reshape(-1,pool.axes).mean(1);mem.extend(ii.tolist());labels.extend([float(readings[read])]*len(ii));mi=np.array(mem);my=np.array(labels)
                engine.refresh(net,np.unique(np.r_[mi,tr] if setting=='A' else mi));net.train()
                for st in range(ONLINE_STEPS):
                    data.CHECK_RESOURCE()
                    aa=np.random.randint(0,len(mi),64);ids=mi[aa];yy=my[aa]
                    if setting=='A':
                        rr=np.random.randint(0,len(tr),64);ids=np.r_[ids,tr[rr]];yy=np.r_[yy,Y[tr[rr]]]
                    sampling_digest.update(ids.tobytes());sampling_digest.update(yy.tobytes())
                    yy=torch.as_tensor(((yy-norm[2])/norm[3]).astype(np.float32))[:,None].to(dev)
                    with torch.autocast(dev.type,dtype=torch.bfloat16,enabled=dev.type=='cuda'):p,_,_=engine.forward(net,ids,True,reconstruct=False);loss=F_loss(p.float(),yy)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite online loss')
                    opt.zero_grad();loss.backward();opt.step()
                if int(read)+1<int(end):
                    ws=np.concatenate(d['groups'][int(read)+1:int(end)]);ids=flatids(ws);raw[ws]=engine.predict(net,ids).reshape(-1,pool.axes).mean(1)
                updates.append(dict(block=int(read),memory_axis_samples=len(mi),latest_allowed_time=float(d['Tb'][read]),memory_last_label=float(readings[read]),steps=ONLINE_STEPS))
            expected=independent_sampling(d,pool,s,tr,N,SEED,ONLINE_STEPS)
            assert sampling_digest.hexdigest()==expected, 'Online sample hash failed independent reconstruction'
            out[tag+'_raw']=raw;out[tag+'_block']=np.array([np.mean(raw[g]) for g in d['groups']]);out[tag]=agg(raw);out['Apos'+str(N) if setting=='A' else 'Bpos'+str(N)]=pos;meta['schedules'][tag]=dict(sampling_sha256=sampling_digest.hexdigest(),updates=updates,noise_sd=sd,start='test' if setting=='A' else 'recording',initial_fitting=setting=='A');print('PERIODIC',s,tag,flush=True)
    np.savez_compressed(path/f'{s}.npz',**out);meta['elapsed_s']=time.time()-t0;done.write_text(json.dumps(meta,indent=2));print('PATIENT_COMPLETE',s,round(meta['elapsed_s']),flush=True)


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


def independent_sampling(d,pool,subject,tr,hours,seed,steps):
    rng=np.random.RandomState(seed);digest=hashlib.sha256();memory=[];labels=[]
    for read in data.schedule(subject,d,hours,'A'):
        ii=pool.ids(subject,d['groups'][int(read)])
        memory.extend(ii.tolist());labels.extend([float(d['Yb'][read])]*len(ii))
        mi=np.asarray(memory);my=np.asarray(labels)
        for _ in range(steps):
            aa=rng.randint(0,len(mi),64);rr=rng.randint(0,len(tr),64)
            ids=np.r_[mi[aa],tr[rr]];yy=np.r_[my[aa],pool.Y[tr[rr]]]
            digest.update(ids.tobytes());digest.update(yy.tobytes())
    return digest.hexdigest()
