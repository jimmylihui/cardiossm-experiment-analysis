"""Existing BCGnet reproduction architecture, extracted without executing data/training code."""
import torch
from torch import nn
class RSU(nn.Module):
    def __init__(s,cin,cmid,cout,L=4):
        super().__init__(); s.L=L
        s.inc=nn.Sequential(nn.Conv1d(cin,cout,3,padding=1),nn.BatchNorm1d(cout),nn.ReLU())
        s.enc=nn.ModuleList(); s.dec=nn.ModuleList()
        for i in range(L): s.enc.append(nn.Sequential(nn.Conv1d(cout if i==0 else cmid,cmid,3,padding=1),nn.BatchNorm1d(cmid),nn.ReLU()))
        s.bot=nn.Sequential(nn.Conv1d(cmid,cmid,3,padding=2,dilation=2),nn.BatchNorm1d(cmid),nn.ReLU())
        for i in range(L):
            co=cout if i==L-1 else cmid; s.dec.append(nn.Sequential(nn.Conv1d(cmid*2,co,3,padding=1),nn.BatchNorm1d(co),nn.ReLU()))
        s.pool=nn.MaxPool1d(2,ceil_mode=True)
    def forward(s,x):
        x=s.inc(x); h=[]; y=x
        for i,e in enumerate(s.enc):
            y=e(y); h.append(y)
            if i<s.L-1: y=s.pool(y)
        y=s.bot(y)
        for i,d in enumerate(s.dec):
            y=d(torch.cat([y,h[-1-i]],1))
            if i<s.L-1: y=nn.functional.interpolate(y,size=h[-2-i].shape[-1],mode='linear',align_corners=False)
        return y+x
class BCGnet(nn.Module):
    architecture_version = "bcgnet_u2_map_v1"
    def __init__(s,nf=42,C=16,nout=1):
        super().__init__()
        s.e=nn.ModuleList([RSU(1,8,C,5),RSU(C,8,C*2,4),RSU(C*2,16,C*4,3),RSU(C*4,16,C*4,3),RSU(C*4,16,C*4,2)])
        s.d=nn.ModuleList([RSU(C*8,16,C*4,3),RSU(C*8,16,C*2,3),RSU(C*4,8,C,4),RSU(C*2,8,C,4)])
        s.pool=nn.MaxPool1d(2,ceil_mode=True); s.out=nn.Conv1d(C,1,3,padding=1)
        s.head=nn.Sequential(nn.Linear(C*4+nf,128),nn.ReLU(),nn.Dropout(0.2),nn.Linear(128,nout))
    def forward(s,x,f):
        h=[]
        for i,e in enumerate(s.e):
            x=e(x); h.append(x)
            if i<len(s.e)-1: x=s.pool(x)
        y=h[-1]; emb=y.mean(-1); z=y
        if not s.training or getattr(s,'prediction_only',False): return s.head(torch.cat([emb,f],1)), None, emb
        for i,d in enumerate(s.d):
            z=nn.functional.interpolate(z,size=h[-2-i].shape[-1],mode='linear',align_corners=False); z=d(torch.cat([z,h[-2-i]],1))
        return s.head(torch.cat([emb,f],1)), s.out(z), emb
