from __future__ import annotations
import argparse,random
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader,Dataset
from evaluate import _read_table,_extract_rt_ratio,_filter_and_normalize_ratios,_build_peak_labels,_write_diff_peaks_txt,_plot_same_diff_judgement,_match_same_pairs_hungarian

LAB2ID={"a":0,"b":1,"c":2,"d":3}; ID2LAB={v:k for k,v in LAB2ID.items()}; EXTS={".csv",".tsv",".txt",".xlsx",".xls"}

def seed_all(seed:int, deterministic:bool=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(bool(deterministic), warn_only=bool(deterministic))
    except TypeError:
        try:
            torch.use_deterministic_algorithms(bool(deterministic))
        except Exception:
            pass
    except Exception:
        pass
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)

def load_profile(path,sheet,rt_col,ratio_col,min_ratio):
    df=_read_table(path,sheet_name=sheet); rt,ratio=_extract_rt_ratio(df,rt_col,ratio_col)
    return _filter_and_normalize_ratios(rt,ratio,min_ratio)

def resample(rt,ratio,n,lo,hi):
    if rt.size==0: return np.zeros((2,n),np.float32)
    idx=np.argsort(rt); rt=rt[idx].astype(float); ratio=ratio[idx].astype(float)
    den=max(1e-8,hi-lo); rt=(rt-lo)/den; ratio=ratio/100.0
    if rt.size==1:
        a=np.full(n,rt[0],np.float32); b=np.full(n,ratio[0],np.float32)
    else:
        xo=np.linspace(0,1,rt.size); xn=np.linspace(0,1,n)
        a=np.interp(xn,xo,rt).astype(np.float32); b=np.interp(xn,xo,ratio).astype(np.float32)
    return np.stack([a,b]).astype(np.float32)

def to_tensor_pair(std_rt,std_ratio,s_rt,s_ratio,n):
    all_rt=np.concatenate([std_rt,s_rt]) if (std_rt.size or s_rt.size) else np.array([0.,1.])
    lo,hi=float(np.min(all_rt)),float(np.max(all_rt))
    return resample(std_rt,std_ratio,n,lo,hi),resample(s_rt,s_ratio,n,lo,hi)

def discover_files(root):
    root=Path(root)
    if not root.exists(): raise FileNotFoundError(root)
    return sorted([str(p) for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTS])

def resolve_train_files(args):
    out={"a":list(args.train_a),"b":list(args.train_b),"c":list(args.train_c),"d":list(args.train_d)}
    if args.train_root:
        for lab in "abcd": out[lab]+=discover_files(Path(args.train_root)/lab)
    extra={"a":args.train_a_dir,"b":args.train_b_dir,"c":args.train_c_dir,"d":args.train_d_dir}
    for lab,folder in extra.items():
        if folder: out[lab]+=discover_files(folder)
    for lab in out: out[lab]=sorted(dict.fromkeys(out[lab]))
    return out

def split_files(grouped,val_ratio,seed):
    rng=random.Random(seed); train={}; val={}
    for lab,files in grouped.items():
        fs=files[:]; rng.shuffle(fs)
        if not fs: raise ValueError(f"No training files for class {lab}.")
        n=max(1,int(round(len(fs)*val_ratio))) if len(fs)>1 and val_ratio>0 else 0
        if n>=len(fs): n=len(fs)-1
        val[lab]=fs[:n]; train[lab]=fs[n:]
        if not train[lab]: train[lab],val[lab]=fs,[ ]
    return train,val

def _resolve_relative_file(path_value,base_dir):
    p=Path(str(path_value))
    if p.is_absolute(): return str(p)
    return str((base_dir/p).resolve())

def load_train_manifest(args):
    if not args.train_manifest: return []
    manifest=Path(args.train_manifest)
    df=_read_table(str(manifest),sheet_name=args.manifest_sheet)
    required={"standard_file","sample_file","label"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"train manifest missing columns: {sorted(missing)}")
    records=[]; base=manifest.parent
    for _,row in df.iterrows():
        lab=str(row["label"]).strip().lower()
        if lab not in LAB2ID: raise ValueError(f"Invalid label in manifest: {lab}. Expected a/b/c/d.")
        std=_resolve_relative_file(row["standard_file"],base); smp=_resolve_relative_file(row["sample_file"],base)
        records.append({"standard_file":std,"sample_file":smp,"label":lab,"name":Path(smp).name})
    if not records: raise ValueError("train manifest contains no valid rows")
    return records

def load_manifest(manifest_path, sheet_name):
    if not manifest_path: return []
    manifest=Path(manifest_path)
    df=_read_table(str(manifest),sheet_name=sheet_name)
    required={"standard_file","sample_file","label"}
    missing=required-set(df.columns)
    if missing: raise ValueError(f"train manifest missing columns: {sorted(missing)}")
    records=[]; base=manifest.parent
    for _,row in df.iterrows():
        lab=str(row["label"]).strip().lower()
        if lab not in LAB2ID: raise ValueError(f"Invalid label in manifest: {lab}. Expected a/b/c/d.")
        std=_resolve_relative_file(row["standard_file"],base); smp=_resolve_relative_file(row["sample_file"],base)
        records.append({"standard_file":std,"sample_file":smp,"label":lab,"name":Path(smp).name})
    if not records: raise ValueError("train manifest contains no valid rows")
    return records

def resolve_val_files(args):
    out={"a":list(args.val_a),"b":list(args.val_b),"c":list(args.val_c),"d":list(args.val_d)}
    if args.val_root:
        for lab in "abcd": out[lab]+=discover_files(Path(args.val_root)/lab)
    extra={"a":args.val_a_dir,"b":args.val_b_dir,"c":args.val_c_dir,"d":args.val_d_dir}
    for lab,folder in extra.items():
        if folder: out[lab]+=discover_files(folder)
    for lab in out: out[lab]=sorted(dict.fromkeys(out[lab]))
    return out

def records_from_grouped(grouped,args):
    if not args.standard_file: raise ValueError("--standard-file is required when training without --train-manifest")
    records=[]
    for lab,files in grouped.items():
        for f in files:
            records.append({"standard_file":args.standard_file,"sample_file":f,"label":lab,"name":Path(f).name})
    return records

def split_records(records,val_ratio,seed):
    rng=random.Random(seed); by_lab={lab:[] for lab in LAB2ID}
    for r in records: by_lab[r["label"]].append(r)
    train=[]; val=[]
    for lab,items in by_lab.items():
        if not items: raise ValueError(f"No training records for class {lab}.")
        fs=items[:]; rng.shuffle(fs)
        n=max(1,int(round(len(fs)*val_ratio))) if len(fs)>1 and val_ratio>0 else 0
        if n>=len(fs): n=len(fs)-1
        val.extend(fs[:n]); train.extend(fs[n:])
        if not fs[n:]: train.extend(fs); val=[]
    rng.shuffle(train); rng.shuffle(val)
    return train,val

class PairDataset(Dataset):
    def __init__(self,std_rt,std_ratio,files_by_label,sheet,rt_col,ratio_col,min_ratio,seq_len):
        self.items=[]
        for lab,files in files_by_label.items():
            for f in files:
                rt,ratio=load_profile(f,sheet,rt_col,ratio_col,min_ratio)
                a,b=to_tensor_pair(std_rt,std_ratio,rt,ratio,seq_len)
                self.items.append((a,b,LAB2ID[lab],f))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        a,b,y,f=self.items[i]
        return torch.from_numpy(a),torch.from_numpy(b),torch.tensor(y),f

class PairRecordDataset(Dataset):
    def __init__(self,records,args):
        self.items=[]
        for r in records:
            std_rt,std_ratio=load_profile(r["standard_file"],args.sheet,args.rt_col,args.ratio_col,args.min_ratio)
            rt,ratio=load_profile(r["sample_file"],args.sheet,args.rt_col,args.ratio_col,args.min_ratio)
            a,b=to_tensor_pair(std_rt,std_ratio,rt,ratio,args.seq_len)
            self.items.append((a,b,LAB2ID[r["label"]],r["sample_file"]))
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        a,b,y,f=self.items[i]
        return torch.from_numpy(a),torch.from_numpy(b),torch.tensor(y),f

class Encoder(nn.Module):
    def __init__(self,seq_len,emb_dim=128,dropout=0.25):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(2,32,5,padding=2),
            nn.ReLU(),
            nn.Conv1d(32,64,5,padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(16),
            nn.Flatten(),
            nn.Linear(64*16,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,emb_dim)
        )
    def forward(self,x): return self.net(x)

class SiameseNet(nn.Module):
    def __init__(self,seq_len,emb_dim=128,dropout=0.25):
        super().__init__()
        self.encoder=Encoder(seq_len,emb_dim,dropout)
        self.head=nn.Sequential(
            nn.Linear(emb_dim*3,128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128,4)
        )
    def forward(self,std_x,smp_x):
        e1=self.encoder(std_x)
        e2=self.encoder(smp_x)
        d=torch.abs(e2-e1)
        logits=self.head(torch.cat([e1,e2,d],dim=1))
        return logits,d

def proto_contrastive(emb,y,temp=0.2):
    if emb.size(0)<2: return emb.new_tensor(0.0)
    emb=nn.functional.normalize(emb,dim=1); sim=(emb@emb.T)/temp; mask=(y[:,None]==y[None,:]).float(); eye=torch.eye(len(y),device=y.device)
    exp=torch.exp(sim)*(1-eye); pos=(exp*mask*(1-eye)).sum(1); den=exp.sum(1)+1e-8; valid=(mask.sum(1)-1)>0
    if not torch.any(valid): return emb.new_tensor(0.0)
    return (-torch.log((pos[valid]+1e-8)/den[valid])).mean()

def accuracy(model,loader,device):
    model.eval(); correct=total=0
    with torch.no_grad():
        for a,b,y,_ in loader:
            a,b,y=a.to(device),b.to(device),y.to(device); pred=model(a,b)[0].argmax(1); correct+=int((pred==y).sum()); total+=int(y.numel())
    return correct/max(1,total)

def plot_training_history(history,save_path):
    epochs=np.arange(1,len(history["loss"])+1)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    axes[0].plot(epochs,history["loss"],color="#C27D52",linewidth=2,label="train_loss")
    if "val_loss" in history and len(history.get("val_loss",[]))==len(epochs):
        axes[0].plot(epochs,history["val_loss"],color="#7D9D6C",linewidth=2,linestyle="--",label="val_loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3,linestyle="--")
    axes[0].legend()
    axes[1].plot(epochs,history["train_acc"],label="train_acc",color="#5B84B1",linewidth=2)
    axes[1].plot(epochs,history["val_acc"],label="val_acc",color="#7D9D6C",linewidth=2)
    axes[1].set_title("Training / Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0,1.02)
    axes[1].grid(alpha=0.3,linestyle="--")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

def train(model,tr_loader,va_loader,args,device):
    counts=np.zeros(4,np.float32)
    for item in tr_loader.dataset.items:
        lbl = item[2]
        counts[int(lbl)] += 1
    w=counts.sum()/np.clip(counts,1,None)
    ce_kwargs = {"weight":torch.tensor(w,dtype=torch.float32)}
    if args.label_smoothing and args.label_smoothing>0.0:
        ce_kwargs["label_smoothing"]=float(args.label_smoothing)
    ce=nn.CrossEntropyLoss(**ce_kwargs).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience, min_lr=args.min_lr, verbose=True)
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=args.reduce_lr_factor, patience=args.reduce_lr_patience, min_lr=args.min_lr)
    best_val_loss=float("inf"); epochs_no_improve=0; best_state=None
    history={"loss":[],"val_loss":[],"train_acc":[],"val_acc":[]}
    for ep in range(args.epochs):
        model.train(); loss_sum=n=0
        for a,b,y,_ in tr_loader:
            a,b,y=a.to(device),b.to(device),y.to(device)
            opt.zero_grad()
            logits,emb=model(a,b)
            loss=ce(logits,y)+args.contrastive_weight*proto_contrastive(emb,y,args.temperature)
            loss.backward(); opt.step()
            loss_sum+=float(loss.item())*y.numel(); n+=int(y.numel())
        avg_loss=loss_sum/max(1,n)
        tr_acc=accuracy(model,tr_loader,device)
        if va_loader:
            va_loss,va_acc = validate(model,va_loader,device,ce,verbose=False)
        else:
            va_loss=avg_loss; va_acc=tr_acc
        scheduler.step(va_loss)
        history["loss"].append(avg_loss); history["val_loss"].append(va_loss); history["train_acc"].append(tr_acc); history["val_acc"].append(va_acc)
        if va_loss+1e-9 < best_val_loss:
            best_val_loss = va_loss
            best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"epoch {ep+1}/{args.epochs} loss = {avg_loss:.6f} val_loss = {va_loss:.6f} train_acc = {tr_acc:.4f} val_acc = {va_acc:.4f} lr = {opt.param_groups[0]['lr']:.6e}")
        if epochs_no_improve >= args.early_stopping_patience:
            print(f"Early stopping: no improvement for {args.early_stopping_patience} epochs.")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    plot_training_history(history,args.save_training_plot)
    print(f"saved_training_plot = {args.save_training_plot}")

def save_ckpt(model,args,mode="single_standard"):
    torch.save({
        "state":model.state_dict(),
        "seq_len":args.seq_len,
        "emb_dim":args.emb_dim,
        "mode":mode,
        "rt_col":args.rt_col,
        "ratio_col":args.ratio_col,
        "min_ratio":args.min_ratio,
        "delta_min":args.delta_min,
        "dropout":getattr(args,"dropout",0.0)
    },args.model_path); print(f"saved_model = {args.model_path}")

def load_ckpt(path,device):
    ck=torch.load(path,map_location=device)
    dropout = float(ck.get("dropout",0.0))
    m=SiameseNet(int(ck["seq_len"]),int(ck["emb_dim"]),dropout).to(device)
    m.load_state_dict(ck["state"]); m.eval(); return m,ck

def predict(model,std_rt,std_ratio,s_rt,s_ratio,seq_len,device):
    a,b=to_tensor_pair(std_rt,std_ratio,s_rt,s_ratio,seq_len)
    ta,tb=torch.from_numpy(a).unsqueeze(0).to(device),torch.from_numpy(b).unsqueeze(0).to(device)
    with torch.no_grad(): p=torch.softmax(model(ta,tb)[0],dim=1).cpu().numpy()[0]
    pred=ID2LAB[int(np.argmax(p))]; return pred,{ID2LAB[i]:float(p[i]) for i in range(4)}

def similarity(std_rt,std_ratio,s_rt,s_ratio,delta):
    i,j=_match_same_pairs_hungarian(std_rt,s_rt,delta)
    same=float(np.sum(np.abs(std_ratio[i]-s_ratio[j]))) if i.size else 0.0
    u1=np.zeros(std_rt.size,bool); u2=np.zeros(s_rt.size,bool); u1[i]=True; u2[j]=True
    result=same+float(np.sum(std_ratio[~u1])+np.sum(s_ratio[~u2]))
    sm1=np.zeros(std_rt.size,dtype=bool); sm2=np.zeros(s_rt.size,dtype=bool)
    sm1[i]=True; sm2[j]=True
    pairs=[(int(x),int(y)) for x,y in zip(i.tolist(),j.tolist())]
    return 100.0-float(result),sm1,sm2,pairs

def ci95(values):
    arr=np.asarray(values,dtype=float)
    if arr.size==0: return {"n":0,"mean":np.nan,"low":np.nan,"high":np.nan,"std":np.nan}
    mean=float(np.mean(arr)); std=float(np.std(arr,ddof=1)) if arr.size>1 else 0.0; sem=std/np.sqrt(arr.size) if arr.size>1 else 0.0
    delta=1.96*sem
    return {"n":int(arr.size),"mean":mean,"low":mean-delta,"high":mean+delta,"std":std}

def collect_similarity_summary_from_records(records,args):
    summary={}; rows=[]
    for lab in ["a","b","c","d"]:
        vals=[]
        for r in records:
            if r["label"] != lab: continue
            std_rt,std_ratio=load_profile(r["standard_file"],args.sheet,args.rt_col,args.ratio_col,args.min_ratio)
            rt,ratio=load_profile(r["sample_file"],args.sheet,args.rt_col,args.ratio_col,args.min_ratio)
            sim,_,_,_=similarity(std_rt,std_ratio,rt,ratio,args.delta_min)
            vals.append(float(sim)); rows.append((lab,Path(r["standard_file"]).name,Path(r["sample_file"]).name,float(sim)))
        summary[lab]={"values":vals,**ci95(vals)}
    return summary,rows

def collect_similarity_summary(grouped,args):
    return collect_similarity_summary_from_records(records_from_grouped(grouped,args),args)

def save_similarity_summary(summary,rows,save_path):
    lines=["group	standard_file	sample_file	similarity"]
    for row in rows:
        if len(row)==4:
            lab,std_name,smp_name,sim=row; lines.append(f"{lab}	{std_name}	{smp_name}	{sim:.6f}")
        else:
            lab,name,sim=row; lines.append(f"{lab}		{name}	{sim:.6f}")
    lines.append("")
    lines.append("group_summary")
    lines.append("group	n	mean	ci_low	ci_high	std")
    for lab in ["a","b","c","d"]:
        s=summary[lab]
        lines.append(f"{lab}	{s['n']}	{s['mean']:.6f}	{s['low']:.6f}	{s['high']:.6f}	{s['std']:.6f}")
    Path(save_path).write_text("\n".join(lines)+"\n",encoding="utf-8")

def _p_to_stars(p):
    if p is None or not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""

def _ttest_1samp_p(values, mu):
    arr=np.asarray(values,dtype=float); arr=arr[np.isfinite(arr)]
    if arr.size < 2: return np.nan
    from scipy import stats
    return float(stats.ttest_1samp(arr, popmean=float(mu)).pvalue)

def plot_sample_classification(prob,summary,sample_similarity,pred_label,save_path):
    labels=["a","b","c","d"]
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    probs=[prob[k] for k in labels]
    colors=["#5B84B1" if k!=pred_label else "#C27D52" for k in labels]
    axes[0].bar(labels,probs,color=colors,alpha=0.9)
    axes[0].set_ylabel("Probability", fontweight="bold")
    axes[0].grid(axis="y",alpha=0.3,linestyle="--")

    means=[summary[k]["mean"] for k in labels]
    lows=[summary[k]["low"] for k in labels]
    highs=[summary[k]["high"] for k in labels]
    err_low=[m-l if np.isfinite(m) and np.isfinite(l) else 0.0 for m,l in zip(means,lows)]
    err_high=[h-m if np.isfinite(m) and np.isfinite(h) else 0.0 for m,h in zip(means,highs)]
    x=np.arange(len(labels))
    axes[1].errorbar(x,means,yerr=[err_low,err_high],fmt="o",capsize=6,color="#5B84B1",linewidth=2,label="train mean ± 95% CI")
    axes[1].axhline(sample_similarity,color="#C27D52",linestyle="--",linewidth=2,label=f"sample similarity = {sample_similarity:.2f}")

    stars_list=[_p_to_stars(_ttest_1samp_p(summary[k].get("values",[]), sample_similarity)) for k in labels]

    for i,stars in enumerate(stars_list):
        if not stars: continue
        axes[0].text(i, float(probs[i])+0.02, stars, ha="center", va="bottom", fontsize=16, color="#C27D52", clip_on=False)
    pmax=max(probs) if probs else 1.0
    axes[0].set_ylim(0, max(1.02, pmax+0.12))

    span_vals=[float(sample_similarity)]+[float(m)+float(e) for m,e in zip(means,err_high) if np.isfinite(m)]
    span=max(span_vals)-min(span_vals) if len(span_vals)>1 else 1.0
    star_pad=max(0.5, span*0.04)
    y_annot=[]
    for i,stars in enumerate(stars_list):
        if not stars: continue
        hi=highs[i] if np.isfinite(highs[i]) else (means[i] if np.isfinite(means[i]) else float(sample_similarity))
        y=float(hi)+star_pad
        y_annot.append(y)
        axes[1].text(x[i], y, stars, ha="center", va="bottom", fontsize=16, color="#C27D52", clip_on=False)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Similarity", fontweight="bold")
    axes[1].grid(axis="y",alpha=0.3,linestyle="--")
    axes[1].legend(loc="best")
    ys=span_vals+y_annot
    if ys:
        y0=min(ys); y1=max(ys); pad=max(1.0,(y1-y0)*0.12 if y1>y0 else 2.0)
        axes[1].set_ylim(y0-pad*0.3, y1+pad*1.6)
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

def plot_similarity_summary(summary,save_path):
    labels=["a","b","c","d"]
    means=[summary[k]["mean"] for k in labels]
    lows=[summary[k]["low"] for k in labels]
    highs=[summary[k]["high"] for k in labels]
    err_low=[m-l if np.isfinite(m) and np.isfinite(l) else 0.0 for m,l in zip(means,lows)]
    err_high=[h-m if np.isfinite(m) and np.isfinite(h) else 0.0 for m,h in zip(means,highs)]
    x=np.arange(len(labels))
    fig,ax=plt.subplots(1,1,figsize=(8,5))
    ax.errorbar(x,means,yerr=[err_low,err_high],fmt='o',capsize=6,color="#5B84B1",linewidth=2)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontweight="bold", fontfamily="Arial", fontsize=14)
    ax.set_ylabel("Similarity", fontweight="bold", fontfamily="Arial", fontsize=14)
    ax.grid(axis="y",alpha=0.3,linestyle="--")
    fig.tight_layout()
    fig.savefig(save_path,dpi=300,bbox_inches="tight")
    plt.close(fig)

def parse_args(argv=None):
    p=argparse.ArgumentParser(description="Siamese PyTorch peptide mapping classifier with folder auto-training.")
    p.add_argument("--standard-file"); p.add_argument("--sample-file")
    p.add_argument("--train-manifest",help="CSV/XLSX manifest for cross-protein training. Required columns: standard_file, sample_file, label.")
    p.add_argument("--manifest-sheet",default="Sheet1",help="Sheet name for XLSX train manifest.")
    p.add_argument("--train-root","--train-dir",dest="train_root",help="Folder containing subfolders a/b/c/d for single-standard training.")
    p.add_argument("--train-a-dir"); p.add_argument("--train-b-dir"); p.add_argument("--train-c-dir"); p.add_argument("--train-d-dir")
    p.add_argument("--train-a",nargs="*",default=[]); p.add_argument("--train-b",nargs="*",default=[]); p.add_argument("--train-c",nargs="*",default=[]); p.add_argument("--train-d",nargs="*",default=[])
    p.add_argument("--val-manifest",help="CSV/XLSX manifest for validation. Required columns: standard_file, sample_file, label.")
    p.add_argument("--val-root","--val-dir",dest="val_root",help="Folder containing subfolders a/b/c/d for single-standard validation.")
    p.add_argument("--val-a-dir"); p.add_argument("--val-b-dir"); p.add_argument("--val-c-dir"); p.add_argument("--val-d-dir")
    p.add_argument("--val-a",nargs="*",default=[]); p.add_argument("--val-b",nargs="*",default=[]); p.add_argument("--val-c",nargs="*",default=[]); p.add_argument("--val-d",nargs="*",default=[])
    p.add_argument("--model-path",default="judge_siamese.pt"); p.add_argument("--sheet",default="Sheet1"); p.add_argument("--rt-col",default="rt"); p.add_argument("--ratio-col",default="ratio")
    p.add_argument("--min-ratio",type=float,default=0.0); p.add_argument("--seq-len",type=int,default=128); p.add_argument("--emb-dim",type=int,default=64)
    p.add_argument("--dropout",type=float,default=0.25,help="Dropout probability for encoder/head (0.2-0.4 recommended)")
    p.add_argument("--label-smoothing",type=float,default=0.0,help="Label smoothing for cross-entropy (0.0 to disable)")
    p.add_argument("--reduce-lr-factor",type=float,default=0.5,help="Factor to reduce LR on plateau")
    p.add_argument("--reduce-lr-patience",type=int,default=3,help="ReduceLROnPlateau patience (epochs)")
    p.add_argument("--min-lr",type=float,default=1e-6,help="Minimum LR after reductions")
    p.add_argument("--early-stopping-patience",type=int,default=10,help="Early stopping patience in epochs")
    p.add_argument("--epochs",type=int,default=300); p.add_argument("--batch-size",type=int,default=16); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--weight-decay",type=float,default=1e-4)
    p.add_argument("--contrastive-weight",type=float,default=0.2); p.add_argument("--temperature",type=float,default=0.2); p.add_argument("--val-ratio",type=float,default=0.25)
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="cuda"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--deterministic",action="store_true",help="Enable strict deterministic algorithms. This may fail on some CUDA ops; disabled by default for GPU training compatibility.")
    p.add_argument("--delta-min",type=float,default=0.208424)
    p.add_argument("--save-plot",default="judge_same_diff_judgement.png"); p.add_argument("--save-diff-txt",default="judge_diff_peaks.txt"); p.add_argument("--save-prediction-txt",default="judge_prediction.txt"); p.add_argument("--save-training-plot",default="judge_training_history.png"); p.add_argument("--save-similarity-summary",default="judge_similarity_summary.txt"); p.add_argument("--save-classification-plot",default="judge_sample_classification.png")
    p.add_argument("--save-similarity-plot",default="judge_group_similarity.png")
    return p.parse_args(argv)

def validate(model, val_loader, device, criterion, verbose=True):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    per_batch_losses = []

    with torch.no_grad():
        for a,b,y,_ in val_loader:
            a = a.to(device); b = b.to(device); y = y.to(device)
            logits, _ = model(a,b)
            loss = criterion(logits, y)
            running_loss += float(loss.item()) * a.size(0)
            per_batch_losses.append(float(loss.item()))
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += a.size(0)

    val_loss = running_loss / max(total, 1)
    val_acc = correct / max(total, 1)
    if verbose:
        mean_loss = float(np.mean(per_batch_losses)) if per_batch_losses else 0.0
        std_loss = float(np.std(per_batch_losses)) if per_batch_losses else 0.0
        print(f"[VAL] loss={val_loss:.6f}, acc={val_acc:.4f}, batch_loss_mean={mean_loss:.6f}, batch_loss_std={std_loss:.6f}")
    return val_loss, val_acc

def main(argv=None):
    args=parse_args(argv); seed_all(args.seed,args.deterministic)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("WARNING: --device cuda requested but CUDA is not available. Falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    train_grouped=resolve_train_files(args); train_manifest_records=load_train_manifest(args)
    train_records=train_manifest_records if train_manifest_records else records_from_grouped(train_grouped,args) if any(train_grouped[k] for k in train_grouped) else []
    do_train=bool(train_records)
    train_mode="cross_protein_manifest" if train_manifest_records else "single_standard"
    val_manifest_records = load_manifest(args.val_manifest,args.manifest_sheet) if getattr(args,'val_manifest',None) else []
    val_grouped = resolve_val_files(args) if any(getattr(args,k) for k in ("val_a","val_b","val_c","val_d","val_root","val_a_dir","val_b_dir","val_c_dir","val_d_dir")) else {"a":[],"b":[],"c":[],"d":[]}
    val_records = val_manifest_records if val_manifest_records else records_from_grouped(val_grouped,args) if any(val_grouped[k] for k in val_grouped) else []
    similarity_summary=None; similarity_rows=[]
    model=None; ck=None
    if do_train:
        tr_records = train_records
        va_records = val_records
        tr_ds=PairRecordDataset(tr_records,args)
        va_ds=PairRecordDataset(va_records,args) if va_records else None
        tr_dl=DataLoader(tr_ds,batch_size=min(args.batch_size,len(tr_ds)),shuffle=True)
        va_dl=DataLoader(va_ds,batch_size=min(args.batch_size,len(va_ds)),shuffle=False) if va_ds and len(va_ds) else None
        model=SiameseNet(args.seq_len,args.emb_dim,args.dropout).to(device); print(f"device = {device}"); print(f"train_mode = {train_mode}"); print(f"train_samples = {len(tr_ds)} val_samples = {0 if va_ds is None else len(va_ds)}")
        train(model,tr_dl,va_dl,args,device); save_ckpt(model,args,train_mode)
        similarity_summary,similarity_rows=collect_similarity_summary_from_records(train_records,args)
        save_similarity_summary(similarity_summary,similarity_rows,args.save_similarity_summary)
        plot_similarity_summary(similarity_summary,args.save_similarity_plot)
        print(f"saved_similarity_plot = {args.save_similarity_plot}")
        print(f"saved_similarity_summary = {args.save_similarity_summary}")
        for lab in ["a","b","c","d"]:
            s=similarity_summary[lab]
            print(f"{lab}_similarity_mean = {s['mean']:.6f}; 95% CI = [{s['low']:.6f}, {s['high']:.6f}]")
    else:
        if not Path(args.model_path).exists(): raise FileNotFoundError("No training data provided and model file not found.")
        model,ck=load_ckpt(args.model_path,device); print(f"loaded_model = {args.model_path}")
    if not args.sample_file: return
    if not args.standard_file: raise ValueError("--standard-file is required for sample prediction")
    if similarity_summary is None and train_records:
        similarity_summary,similarity_rows=collect_similarity_summary_from_records(train_records,args)
        save_similarity_summary(similarity_summary,similarity_rows,args.save_similarity_summary)
        plot_similarity_summary(similarity_summary,args.save_similarity_plot)
        print(f"saved_similarity_plot = {args.save_similarity_plot}")
        print(f"saved_similarity_summary = {args.save_similarity_summary}")
    if similarity_summary is None:
        similarity_summary={lab:{"n":0,"mean":np.nan,"low":np.nan,"high":np.nan,"std":np.nan,"values":[]} for lab in ["a","b","c","d"]}
    if model is None: model,ck=load_ckpt(args.model_path,device)
    seq_len=args.seq_len if ck is None else int(ck["seq_len"])
    std_rt,std_ratio=load_profile(args.standard_file,args.sheet,args.rt_col,args.ratio_col,args.min_ratio); s_rt,s_ratio=load_profile(args.sample_file,args.sheet,args.rt_col,args.ratio_col,args.min_ratio)
    pred,prob=predict(model,std_rt,std_ratio,s_rt,s_ratio,seq_len,device)
    sim,sm1,sm2,pairs=similarity(std_rt,std_ratio,s_rt,s_ratio,args.delta_min)
    n1,n2=Path(args.standard_file).stem,Path(args.sample_file).stem
    l1=_build_peak_labels(n1,std_rt); l2=_build_peak_labels(n2,s_rt)
    _write_diff_peaks_txt(args.save_diff_txt,n1,n2,l1,l2,std_rt,s_rt,std_ratio,s_ratio,sm1,sm2)
    _plot_same_diff_judgement(std_rt,s_rt,n1,n2,l1,l2,sm1,sm2,pairs,sim,args.save_plot)
    lines=[
        f"sample_file = {args.sample_file}",
        f"standard_file = {args.standard_file}",
        f"predicted_class = {pred}",
        f"similarity = {sim:.6f}",
        "class_probabilities:",
    ]+[f"  {k}: {prob[k]:.6f}" for k in ["a","b","c","d"]]
    Path(args.save_prediction_txt).write_text("\n".join(lines)+"\n",encoding="utf-8")
    plot_sample_classification(prob,similarity_summary,sim,pred,args.save_classification_plot)
    print(f"predicted_class = {pred}")
    print(f"similarity = {sim:.6f}")
    for k in ["a","b","c","d"]: print(f"prob_{k} = {prob[k]:.6f}")
    print(f"saved_prediction_txt = {args.save_prediction_txt}")
    print(f"saved_diff_txt = {args.save_diff_txt}")
    print(f"saved_plot = {args.save_plot}")
    print(f"saved_classification_plot = {args.save_classification_plot}")

if __name__=="__main__":
    main()
