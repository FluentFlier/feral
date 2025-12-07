import os
import json
import re
import math
import argparse
from pathlib import Path

DEFAULT_FPS = 30.0
TRAIN_FRAC = 0.7
VAL_FRAC   = 0.15
TEST_FRAC  = 0.15

def _norm_name(s):
    return re.sub(r"\s+", "_", str(s).strip().lower())

def _safe_float(x, default=None):
    if x is None: return default
    if isinstance(x, str) and ";" in x: x = x.split(";", 1)[0]
    try: return float(x)
    except: return default

def _read_boris_records(path):
    p = Path(path)
    ext = p.suffix.lower()
    delim = "," if ext == ".csv" else "\t"
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx, header_cols = None, None
    for i, raw in enumerate(lines):
        s = raw.strip()
        if not s: continue
        parts = [x.strip() for x in s.split(delim)]
        low = " ".join(x.lower() for x in parts)
        if ("time" in low and "behavior" in low and "fps" in low):
            header_idx, header_cols = i, parts
            break
        if ("time" in low and "media file path" in low and "status" in low):
            header_idx, header_cols = i, parts
            break

    if header_idx is not None:
        rows = []
        for raw in lines[header_idx+1:]:
            if not raw.strip(): continue
            parts = raw.strip("\n").split(delim)
            if len(parts) < len(header_cols): parts += [""] * (len(header_cols)-len(parts))
            if len(parts) > len(header_cols): parts = parts[:len(header_cols)]
            rows.append(parts)
        return header_cols, [dict(zip(header_cols, r)) for r in rows]
    
    # Fallback
    fb = ["Time","Media file path","Total length","FPS","Subject","Behavior","Behavioral category","Comment","Status"]
    rows=[]
    for raw in lines:
        if not raw.strip(): continue
        parts=raw.strip("\n").split(delim)
        t=_safe_float(parts[0],None)
        if t is None: continue
        if len(parts)<len(fb): parts+=[""]*(len(fb)-len(parts))
        if len(parts)>len(fb): parts=parts[:len(fb)]
        rows.append(parts)
    return fb, [dict(zip(fb,r)) for r in rows]

def _find_col(hdr, name):
    name=name.lower()
    for c in hdr:
        if c.lower()==name: return c
    for c in hdr:
        if name in c.lower(): return c
    return None

def _detect_status_col(hdr, recs):
    if not hdr or not recs: return None
    cand=[c for c in hdr if any(k in c.lower() for k in ["status","comment","event","state"])]
    def has(col):
        for r in recs:
            v=str(r.get(col,"")).lower()
            if "start" in v or "stop" in v: return True
        return False
    for c in cand:
        if has(c): return c
    for c in hdr:
        if has(c): return c
    return None

def _video_name(records, col_path, col_name, fallback):
    media=None
    for col in [col_path, col_name]:
        if col:
            for r in records:
                val=r.get(col,"").strip()
                if val:
                    media=val
                    break
    if media: name=Path(media).name
    else: name=f"{fallback}.mp4"
    if "." not in name: name=f"{name}.mp4"
    return name

def _fps_and_duration(recs, col_fps, col_total, col_time):
    fps=None; total=None
    for r in recs:
        if fps is None and col_fps: fps=_safe_float(r.get(col_fps),None)
        if total is None and col_total: total=_safe_float(r.get(col_total),None)
        if fps and total: break
    if fps is None or fps<=0 or not math.isfinite(fps): fps=DEFAULT_FPS
    if total is None or total<=0 or not math.isfinite(total):
        mx=0.0
        for r in recs:
            t=_safe_float(r.get(col_time),None)
            if t and t>mx: mx=t
        total=mx
    return fps,total

def _collect_all_behaviors(paths):
    beh=set()
    for path in paths:
        hdr,recs=_read_boris_records(path)
        c=_find_col(hdr,"Behavior")
        if not c: continue
        for r in recs:
            b=str(r.get(c,"")).strip()
            if b: beh.add(_norm_name(b))
    return sorted(beh)

def _make_splits(videos):
    uniq=[]
    seen=set()
    for v in videos:
        if v not in seen:
            seen.add(v); uniq.append(v)
    videos=uniq
    n=len(videos)
    if n==0: return {"train":[],"val":[],"test":[],"inference":[]}
    n_train=max(1,int(round(n*TRAIN_FRAC)))
    n_val=int(round(n*VAL_FRAC))
    if n_train+n_val>n: n_val=max(0,n-n_train)
    n_test=max(0,n-n_train-n_val)
    return {
        "train":videos[:n_train],
        "val":videos[n_train:n_train+n_val],
        "test":videos[n_train+n_val:n_train+n_val+n_test],
        "inference":videos
    }

def _labels_from_records_single(recs,fps,frames,col_time,col_beh,col_status,beh_to_id):
    frames=max(0,int(frames))
    if frames==0: return []
    out=[0]*frames
    act={}
    def tf(t):
        if t is None: return None
        f=int(round(t*fps))
        return max(0,min(f,frames-1))
    for r in recs:
        t=_safe_float(r.get(col_time),None)
        b=str(r.get(col_beh,"")).strip()
        if not b or t is None: continue
        beh=_norm_name(b)
        cid=beh_to_id.get(beh,0)
        f=tf(t)
        if f is None: continue
        st=str(r.get(col_status,"")).upper()
        if "START" in st: act[beh]=f
        elif "STOP" in st:
            if beh in act:
                s=act.pop(beh)
                a=min(s,f); b=max(s,f)
                a=max(0,a); b=min(frames-1,b)
                for ff in range(a,b+1): out[ff]=cid
        else: out[f]=cid
    return out

def _labels_from_records_multilabel(recs,fps,frames,col_time,col_beh,col_status,beh_to_id,K):
    frames=max(0,int(frames))
    if frames==0: return []
    out=[[0]*K for _ in range(frames)]
    act={}
    def tf(t):
        if t is None: return None
        f=int(round(t*fps))
        return max(0,min(f,frames-1))
    for r in recs:
        t=_safe_float(r.get(col_time),None)
        raw=str(r.get(col_beh,"")).strip()
        if not raw or t is None: continue
        beh=_norm_name(raw)
        if beh not in beh_to_id: continue
        cid=beh_to_id[beh]
        f=tf(t)
        if f is None: continue
        st=str(r.get(col_status,"")).upper()
        if "START" in st: act[beh]=f
        elif "STOP" in st:
            if beh in act:
                s=act.pop(beh)
                a=min(s,f); b=max(s,f)
                a=max(0,a); b=min(frames-1,b)
                for ff in range(a,b+1): out[ff][cid]=1
        else: out[f][cid]=1
    return out

def convert_labels(input_path, mode="single"):
    p=Path(input_path)
    if p.is_dir():
        files=sorted(list(p.glob("*.tsv"))+list(p.glob("*.csv")))
        out_json=p/"feral_behavioral_labels.json"
    else:
        files=[p]
        out_json=p.with_name("feral_behavioral_labels.json")
    
    if not files:
        print(f"❌ No CSV/TSV files found in {input_path}")
        return

    beh=_collect_all_behaviors(files)
    if not beh:
        print("❌ No behaviors found.")
        return

    if mode == "multilabel":
        beh_to_id={b:i for i,b in enumerate(beh)}
        K=len(beh)
    else:
        beh_to_id={"other":0}
        i=1
        for b in beh:
            beh_to_id[b]=i; i+=1
    
    labels={}
    vids=[]
    for f in files:
        hdr,recs=_read_boris_records(f)
        col_time=_find_col(hdr,"Time")
        col_path=_find_col(hdr,"Media file path")
        col_name=_find_col(hdr,"Media file name")
        col_total=_find_col(hdr,"Total length") or _find_col(hdr,"Media duration")
        col_fps=_find_col(hdr,"FPS")
        col_beh=_find_col(hdr,"Behavior")
        col_status=_detect_status_col(hdr,recs)
        fps,total=_fps_and_duration(recs,col_fps,col_total,col_time)
        frames=int(round(fps*total))
        vname=_video_name(recs,col_path,col_name,f.stem)
        vids.append(vname)
        
        if mode == "multilabel":
            labels[vname]=_labels_from_records_multilabel(recs,fps,frames,col_time,col_beh,col_status,beh_to_id,K)
        else:
            labels[vname]=_labels_from_records_single(recs,fps,frames,col_time,col_beh,col_status,beh_to_id)
        print(f"• {vname} | frames={frames}")

    splits=_make_splits(vids)
    out={
        "is_multilabel": (mode == "multilabel"),
        "class_names": {str(v):k for k,v in beh_to_id.items()} if mode=="single" else {str(i):b for b,i in beh_to_id.items()},
        "labels":labels,
        "splits":splits
    }
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump(out,f,separators=(",",":"),ensure_ascii=False)
    print(f"✅ Saved labels to: {out_json}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path")
    parser.add_argument("--mode", default="single")
    args = parser.parse_args()
    convert_labels(args.input_path, args.mode)
