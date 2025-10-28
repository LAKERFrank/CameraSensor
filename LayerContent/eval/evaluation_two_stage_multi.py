#!/usr/bin/env python3
# evaluation_two_stage.py  –  serve-dead lock, then hit evaluation
from __future__ import annotations
import argparse, json, sys
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from math import inf

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False

from lib.common import ROOTDIR

BIG = 1e9   # huge cost for impossible pairs

LABEL_RE = re.compile(r"label_(\d{8}_\d{4})\.json$")

def find_latest_label(folder: Path) -> Path | None:
    """
    回傳 folder 中時間戳最大的 label_YYYYMMDD_HHMM.json 檔案 Path。
    若找不到符合格式的檔案則傳回 None。
    """
    cand: list[tuple[datetime, Path]] = []
    for p in folder.glob("label_*.json"):
        m = LABEL_RE.match(p.name)
        if not m:
            continue
        try:
            ts = datetime.strptime(m.group(1), "%Y%m%d_%H%M")
            cand.append((ts, p))
        except ValueError:
            pass   # 無效時間字串 → 跳過
    if not cand:
        return None
    # 取日期時間最大的那筆
    return max(cand, key=lambda t: t[0])[1]

# ──────────────────  JSON 讀取  ──────────────────
def load_events(path: Path) -> List[Dict]:
    """flatten events / map touch→hit / skip rest-hit|dead"""
    with path.open() as f: data = json.load(f)
    ev = []
    for ph in data:
        ptype = ph.get("phase")
        for it in ph.get("description", []):
            if it.get("type") != "event":
                continue
            et = it["event_type"]
            if ptype == "rest" and et in {"rest-hit", "rest-dead"}:
                continue
            if et == "touch": 
                et = "hit"
            ev.append({"type": et, "ts": it["timestamp"], "fid": it.get("fid")})
    return ev

def load_rallies(path: Path) -> List[Dict]:
    """return list: {serve_ts, dead_ts, hits:[{ts,fid},…]}"""
    with path.open() as f:
        data = json.load(f)
    out=[]
    for ph in data:
        if ph.get("phase")!="rally":
            continue
        s_ts=d_ts=None
        hits=[]
        for it in ph.get("description", []):
            if it.get("type")!="event":
                continue
            # et="hit" if it["event_type"]=="touch" else it["event_type"]
            et=it["event_type"]
            if et=="serve":
                s_ts=it["timestamp"]
                s_fid = it["fid"]
            elif et=="dead":
                d_ts=it["timestamp"]
                d_fid = it["fid"]
            elif et=="touch":
                hits.append({"ts":it["timestamp"],"fid":it.get("fid"), "kind":"touch"})
            elif et=="hit":
                hits.append({"ts":it["timestamp"],"fid":it.get("fid"), "kind":"hit"})
        out.append({"s_ts":s_ts,"s_fid":s_fid,"d_ts":d_ts,"d_fid":d_fid,"hits":hits})
    return out

# ──────────────────  Greedy/Hungarian 指派  ──────────────────
def greedy_match(gt, pr, tol, same_type=True):
    used=set(); pairs=[]; fn=[]
    for g in gt:
        cand=[(j,p,abs(g["ts"]-p["ts"])) for j,p in enumerate(pr)
              if j not in used and abs(g["ts"]-p["ts"])<=tol]
        if cand:
            j,p,dt=min(cand,key=lambda x:x[2]); used.add(j)
            pairs.append((g,p,dt))
        else: fn.append(g)
    fp=[p for j,p in enumerate(pr) if j not in used]
    return pairs,fp,fn

# ──────────────────  Stage A：serve-dead  ──────────────────
# def eval_rally_ServeDead(gt_path:Path, pr_path:Path, tol:float):
#     gt_r, pr_r = load_rallies(gt_path), load_rallies(pr_path)
#     full, part, fp, fn = [], [], [], []
#     matched=set()
#     for pr in pr_r:
#         cand=[(i,g) for i,g in enumerate(gt_r)
#               if i not in matched and abs(g["s_ts"]-pr["s_ts"])<=tol]
#         if not cand:
#             fp.append(pr)
#             continue
#         i,g=min(cand,key=lambda x:abs(x[1]["s_ts"]-pr["s_ts"]))
#         matched.add(i)
#         dead_ok=abs(g["d_ts"]-pr["d_ts"])<=tol if g["d_ts"] and pr["d_ts"] else False
#         (full if dead_ok else part).append({"gt":g,"pred":pr})
#     fn=[g for i,g in enumerate(gt_r) if i not in matched]
#     return full, part, fp, fn
def eval_rally_ServeDead(gt_path: Path, pr_path: Path, tol_serve: float, tol_dead: float):
    """Pair predicted rallies to GT using BOTH serve & dead timestamps.

    Returns:
        full  – list of dict{gt, pred}
        part  – list of dict{gt, pred}
        fp    – list of unmatched pred rallies
        fn    – list of unmatched gt  rallies
    """
    gt_r, pr_r = load_rallies(gt_path), load_rallies(pr_path)
    m, n = len(gt_r), len(pr_r)

    # ── 1. 建成本矩陣 ───────────────────────────────────
    cost = [[BIG for _ in range(n)] for _ in range(m)]

    for i, g in enumerate(gt_r):
        for j, p in enumerate(pr_r):
            ds = abs(g["s_ts"] - p["s_ts"]) if g["s_ts"] and p["s_ts"] else inf
            dd = abs(g["d_ts"] - p["d_ts"]) if g["d_ts"] and p["d_ts"] else inf
            best = min(ds, dd)
            # if best <= tol:                   # 任一端在容忍度內才算候選
            #     cost[i][j] = best             # 成本 = 較小的 Δt
            if ds <= tol_serve or dd <= tol_dead:   # 任一端符合容忍度
                cost[i][j] = min(ds, dd)

    # ── 2. 指派：Hungarian 若可用，否則 Greedy ───────────
    try:
        row_ind, col_ind = linear_sum_assignment(cost)
    except ImportError:
        # Greedy fallback：逐 GT 取最小成本
        row_ind, col_ind = [], []
        used_pred = set()
        for i, row in enumerate(cost):
            # 找可用且成本最小
            cand = [(j, c) for j, c in enumerate(row) if j not in used_pred and c < BIG]
            if cand:
                j, _ = min(cand, key=lambda x: x[1])
                row_ind.append(i); col_ind.append(j)
                used_pred.add(j)

    # ── 3. 分類 full / part / fp / fn ──────────────────
    full, part, matched_gt, matched_pr = [], [], set(), set()

    for r, c in zip(row_ind, col_ind):
        if cost[r][c] >= BIG:                # 不可配行列
            continue
        g, p = gt_r[r], pr_r[c]
        matched_gt.add(r)
        matched_pr.add(c)

        serve_ok = abs(g["s_ts"] - p["s_ts"]) <= tol_serve if g["s_ts"] and p["s_ts"] else False
        dead_ok  = abs(g["d_ts"] - p["d_ts"]) <= tol_dead if g["d_ts"] and p["d_ts"] else False
        (full if serve_ok and dead_ok else part).append({"gt": g, "pred": p})

    fp = [pr_r[j] for j in range(n) if j not in matched_pr]
    fn = [gt_r[i] for i in range(m) if i not in matched_gt]
    return full, part, fp, fn

# ──────────────────  Stage B：hit  ──────────────────
# def eval_hits(full_rallies: list, tol_hit: float):
#     """Return global TP/FP/FN ＋ 每回合明細 for verbose printing"""
#     global_tp = global_fp = global_fn = 0
#     per_rally = []                       # ← 新增

#     for idx, r in enumerate(full_rallies, 1):
#         g_hits = r["gt"]["hits"]
#         p_hits = r["pred"]["hits"]
#         pairs, fp, fn = greedy_match(g_hits, p_hits, tol_hit, same_type=True)

#         per_rally.append({               # 保留配對明細
#             "id": idx,
#             "serve_ts": r["gt"]["s_ts"],
#             "serve_fid": r["gt"]["s_fid"],
#             "dead_ts":  r["gt"]["d_ts"],
#             "dead_fid": r["gt"]["d_fid"],
#             "pairs": pairs,
#             "fp": fp,
#             "fn": fn,
#         })

#         global_tp += len(pairs)
#         global_fp += len(fp)
#         global_fn += len(fn)

#     return (global_tp, global_fp, global_fn, per_rally)
def eval_hits(full_rallies: list, tol_hit: float, tol_touch: float,
              allow_sub: bool, max_gap: float, sub_cost: float):
    """
    Returns:
        tp_corr, tp_sub, fp, fn, detail[]
        detail: [{id, serve_ts, dead_ts, pairs[(g,p,dt,tag)], fp, fn}, ...]
    tag = 'tp' | 'sub'
    """
    tp_corr = tp_sub = fp_tot = fn_tot = 0
    detail = []

    for idx, r in enumerate(full_rallies, 1):
        g_hits, p_hits = r["gt"]["hits"], r["pred"]["hits"]

        # ── 建成本矩陣 ──────────────────────────
        m, n = len(g_hits), len(p_hits)
        C = [[BIG]*n for _ in range(m)]
        for i,g in enumerate(g_hits):
            tol_evt = tol_touch if g.get("kind") == "touch" else tol_hit
            for j,p in enumerate(p_hits):
                dt = abs(g["ts"]-p["ts"])
                if dt <= tol_evt:        # 正常 TP
                    C[i][j] = dt
                elif allow_sub and dt <= max_gap:      # substitution
                    C[i][j] = sub_cost

        # ── 指派：Hungarian / Greedy fallback ──
        pairs=[]
        matched_p=set()
        matched_g=set()
        if SCIPY_OK and m and n:
            row,col = linear_sum_assignment(C)
            for r_i,c_j in zip(row,col):
                if C[r_i][c_j] >= BIG: continue
                g,p=g_hits[r_i],p_hits[c_j]
                dt=abs(g["ts"]-p["ts"])
                tag='tp' if dt<=tol_evt else 'sub'
                pairs.append((g,p,dt,tag))
                matched_p.add(c_j); matched_g.add(r_i)
        else:
            # Greedy fallback
            for i,g in enumerate(g_hits):
                cand=[(j,p,abs(g["ts"]-p["ts"])) for j,p in enumerate(p_hits)
                      if j not in matched_p and abs(g["ts"]-p["ts"])<=max_gap]
                if not cand: continue
                j,p,dt=min(cand,key=lambda x:x[2])
                tag='tp' if dt<=tol_evt else 'sub'
                pairs.append((g,p,dt,tag))
                matched_p.add(j); matched_g.add(i)

        fp=[p_hits[j] for j in range(n) if j not in matched_p]
        fn=[g_hits[i] for i in range(m) if i not in matched_g]

        tp_corr += sum(1 for *_,t in pairs if t=='tp')
        tp_sub  += sum(1 for *_,t in pairs if t=='sub')
        fp_tot  += len(fp)
        fn_tot  += len(fn)

        detail.append({
            "id": idx,
            "serve_ts": r["gt"]["s_ts"], "serve_fid": r["gt"]["s_fid"],
            "dead_ts":  r["gt"]["d_ts"], "dead_fid":  r["gt"]["d_fid"],
            "pairs": pairs, "fp": fp, "fn": fn
        })

    return tp_corr, tp_sub, fp_tot, fn_tot, detail

# ──────────────────  CLI & main  ──────────────────
def pct(x):
    return f"{x*100:5.2f}%"

def safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0

def args():
    ap=argparse.ArgumentParser("two-stage evaluator")
    ap.add_argument("-d", "--date", nargs="+", required=True, help="一個或多個日期資料夾")
    # ap.add_argument("--tol", type=float, default=0.10, help="serve/dead tol (s)")
    ap.add_argument("--tol-serve", type=float, default=0.10, help="Time tolerance (s) for serve events")
    ap.add_argument("--tol-dead", type=float, default=0.10, help="Time tolerance (s) for dead events")
    ap.add_argument("--tol-hit", type=float, default=0.10, help="hit tol (s)")
    ap.add_argument("--tol-touch", type=float, default=0.10, help="Time tolerance (s) specifically for GT 'touch' events")
    ap.add_argument("-v", "--verbose", action="store_true", help="Print detailed match lists")
    ap.add_argument("--allow-sub", action="store_true", help="啟用 substitution（dt ≤ max-gap 仍算 TP，計為 SUB）")
    ap.add_argument("--hit-max-gap", type=float, default=0.30)
    ap.add_argument("--hit-sub-cost", type=float, default=1.0)
    return ap.parse_args()

def evaluate_one(date_str: str, a: argparse.Namespace) -> Tuple[Dict, Dict]:
    """
    依 date_str 讀檔並計算 Stage-A & Stage-B 指標
    返回 (stageA_counts, stageB_counts)
    stageA_counts keys = full, part, fp, fn
    stageB_counts keys = tp_c, tp_sub, fp, fn
    """
    folder = Path(ROOTDIR) / "replay" / date_str
    if not folder.exists():
        print(f"[WARN] 路徑不存在: {folder}")
        return None, None

    # ------------ 讀取 GT 與 PRED 檔案 ------------
    gt_p = folder / "3D_event_labeled.json"
    pr_p = folder / "label_20250821_2325.json"
    if not pr_p.exists():                       # 若無 label.json → 找最新 label_*.json
        alt = find_latest_label(folder)
        if not alt:
            print(f"[WARN] {folder} 找不到預測檔案")
            return None, None
        pr_p = alt
        print(f"[INFO] {date_str} → 使用 {alt.name}")
    if not gt_p.exists():
        print(f"[WARN] {folder} 找不到 GT 檔案"); return None, None
    
    # ------------ Stage-A ------------
    full, part, fp_r, fn_r = eval_rally_ServeDead(gt_p, pr_p, a.tol_serve, a.tol_dead)

    sa_full, sa_part, sa_fp, sa_fn = map(len, (full, part, fp_r, fn_r))
    P_a = safe_div(2*sa_full + sa_part, 2*(sa_full + sa_part + sa_fp))
    R_a = safe_div(2*sa_full + sa_part, 2*(sa_full + sa_part + sa_fn))
    F1_a = safe_div(2*P_a*R_a, P_a+R_a) if (P_a+R_a) else 0
    A_a  = safe_div(2*sa_full + sa_part,
                    2*(sa_full + sa_part + sa_fp + sa_fn))
    stageA = dict(full=full, part=part, fp=fp_r, fn=fn_r,
                  P=P_a, R=R_a, F1=F1_a, A=A_a)
    
    # 📌 rally-level Stage-A -––––––––––––––––––
    rally_level_A=[]
    def _event_PRF(g_has,p_has,ok):
        tp=1 if g_has and p_has and ok else 0
        fp=1 if (p_has and not ok) or (p_has and not g_has) else 0
        fn=1 if (g_has and not ok) or (g_has and not p_has) else 0
        return tp,fp,fn
    for r in (full+part+
              [{"gt":None,"pred":x} for x in fp_r]+
              [{"gt":x,"pred":None} for x in fn_r]):
        g,p=r.get("gt"),r.get("pred")
        # serve
        g_s,p_s = bool(g and g["s_ts"]), bool(p and p["s_ts"])
        ok_s = g_s and p_s and abs(g["s_ts"]-p["s_ts"])<=a.tol_serve
        tp_s,fp_s,fn_s=_event_PRF(g_s,p_s,ok_s)
        # dead
        g_d,p_d = bool(g and g["d_ts"]), bool(p and p["d_ts"])
        ok_d = g_d and p_d and abs(g["d_ts"]-p["d_ts"])<=a.tol_dead
        tp_d,fp_d,fn_d=_event_PRF(g_d,p_d,ok_d)
        rally_level_A.append((tp_s+tp_d , fp_s+fp_d , fn_s+fn_d))

    stageA=dict(full=full,part=part,fp=fp_r,fn=fn_r,
                P=P_a,R=R_a,F1=F1_a,A=A_a)
    
    # ------------ Stage-B ------------
    def _serve_ts(r):
    # 有 gt 時用 GT serve；只有 pred 時（罕見）退而用 pred serve
        return r["gt"]["s_ts"] if r.get("gt") else r["pred"]["s_ts"]
    
    rallies_for_hit = sorted(full + part, key=_serve_ts)
    tp_c, tp_sub, fp_h, fn_h, detail = eval_hits(
        rallies_for_hit,
        tol_hit=a.tol_hit,
        tol_touch=a.tol_touch,
        allow_sub=a.allow_sub,
        max_gap=a.hit_max_gap,
        sub_cost=a.hit_sub_cost
    )

    eff_tp = tp_c
    P_b = safe_div(eff_tp, eff_tp + fp_h + tp_sub)
    R_b = safe_div(eff_tp, eff_tp + fn_h + tp_sub)
    F1_b = safe_div(2*P_b*R_b, P_b+R_b) if (P_b+R_b) else 0

    stageB = dict(tp_c=tp_c, tp_sub=tp_sub, fp=fp_h, fn=fn_h,
                  P=P_b, R=R_b, F1=F1_b, detail=detail)
    
    # 📌 rally-level Stage-B -––––––––––––––––––
    rally_level_B=[]
    for rd in detail:
        eff_tp_r = sum(1 for *_,tag in rd["pairs"] if tag=='tp')
        sub_tp_r = sum(1 for *_,tag in rd["pairs"] if tag=='sub')
        fp_rly   = len(rd["fp"])
        fn_rly   = len(rd["fn"])
        rally_level_B.append((eff_tp_r, fp_rly+sub_tp_r, fn_rly+sub_tp_r))

    return stageA, stageB, rally_level_A, rally_level_B

def _macro_from_rallies(rallies:List[Tuple[int,int,int]])->Tuple[float,float,float]:
    if not rallies: return 0.0,0.0,0.0
    P=np.mean([tp/(tp+fp) if (tp+fp) else 0 for tp,fp,fn in rallies])
    R=np.mean([tp/(tp+fn) if (tp+fn) else 0 for tp,fp,fn in rallies])
    F1=np.mean([(2*tp)/(2*tp+fp+fn) if (2*tp+fp+fn) else 0
                for tp,fp,fn in rallies])
    return P,R,F1

def main():
    a = args()
    dates = a.date
    if not dates:
        print("必須至少提供一個日期")
        sys.exit(1)
    
    per_date_A, per_date_B = [], []
    per_rally_A_all, per_rally_B_all=[] ,[]


    for d in dates:
        SA, SB, rA, rB = evaluate_one(d, a)
        if SA is None:
            continue

        per_date_A.append(SA)
        per_date_B.append(SB)
        per_rally_A_all.extend(rA)
        per_rally_B_all.extend(rB)

        if a.verbose:
            print(f"\n─── {d} ────────────────────────────────────────")
            full, part, fp_r, fn_r = SA['full'], SA['part'], SA['fp'], SA['fn']
            P_r, R_r, F1_r, A_r = SA['P'], SA['R'], SA['F1'], SA['A']
            print("Stage-A  Serve-Dead")
            print(f"Full {len(full)}  Part {len(part)}  FP {len(fp_r)}  FN {len(fn_r)}")
            print(f"P {pct(P_r)}  R {pct(R_r)}  F1 {pct(F1_r)} A {pct(A_r)}")

            def _show_rally(tag: str, data: Dict, tol_serve: float=a.tol_serve, tol_dead: float=a.tol_dead):
                """
                tag : 'FULL' | 'PART' | 'FP' | 'FN'
                data: {"gt": {...}|None, "pred": {...}|None}
                """
                g, p = data.get("gt"), data.get("pred")

                def fmt(side):
                    if not side:
                        return "—"
                    return f"fid={side['fid']}, ts={side['ts']:.3f}"

                # ── serve / dead 基本資訊 ─────────────────────────────
                g_s  = g and {"fid": g["s_fid"], "ts": g["s_ts"]}
                p_s  = p and {"fid": p["s_fid"], "ts": p["s_ts"]}
                g_d  = g and {"fid": g["d_fid"], "ts": g["d_ts"]}
                p_d  = p and {"fid": p["d_fid"], "ts": p["d_ts"]}

                # ── 計算時間差（若其中一方缺就 None） ─────────────────
                dt_s = abs(g_s["ts"] - p_s["ts"]) if g_s and p_s else None
                dt_d = abs(g_d["ts"] - p_d["ts"]) if g_d and p_d else None

                # ── 決定哪端失配 ─────────────────────────────────────
                reason = ""
                if tag == "PART":
                    if dt_s is not None and dt_s > tol_serve:
                        reason = f"serve Δt={dt_s:.3f}s"
                    elif dt_d is not None and dt_d > tol_dead:
                        reason = f"dead Δt={dt_d:.3f}s"
                    else:
                        reason = "serve OK, dead missing"
                elif tag == "FP":
                    reason = "no matching GT serve"
                elif tag == "FN":
                    reason = "no matching PRED serve"

                # ── 列印一行 ──────────────────────────────────────────
                print(
                    f"{tag:<5} "
                    f"GT[serve {fmt(g_s)} | dead {fmt(g_d)}]   ↔   "
                    f"PRED[serve {fmt(p_s)} | dead {fmt(p_d)}]   "
                    f"{'(' + reason + ')' if reason else ''}"
                )

            print("\n── Serve-Dead rally details ──")
            for r in full:
                _show_rally("FULL", r)
            for r in part:
                _show_rally("PART", r)
            for r in fp_r:
                _show_rally("FP", {"gt": None, "pred": r})
            for r in fn_r:
                _show_rally("FN", {"gt": r, "pred": None})

            print("\n" + "*" * 150)

            tp_c, tp_sub, fp_h, fn_h = SB['tp_c'], SB['tp_sub'], SB['fp'], SB['fn']
            P_h, R_h, F1_h = SB['P'], SB['R'], SB['F1']
            hit_detail = SB['detail']
            print("\nStage-B  Hit  (within Full & Partial rallies)")
            print(f"TP-corr {tp_c}  SUB {tp_sub}  FP {fp_h}  FN {fn_h}")
            print(f"P {pct(P_h)}  R {pct(R_h)}  F1 {pct(F1_h)}")
            print("\n── Hit matchup details ──")
            for rd in hit_detail:
                print(f"\n[Rally {rd['id']}] Serve fid={rd['serve_fid']} ts={rd['serve_ts']:.3f}s  →  Dead fid={rd['dead_fid']} ts={rd['dead_ts']:.3f}s")
                for g, p, dt, tag in rd["pairs"]:
                    label = "Real TP" if tag == "tp" else "Sub TP"
                    print(f"[{label:<3}] "
                        f"GT(fid={g['fid']} ts={g['ts']:.3f}) "
                        f"↔ PRED(fid={p['fid']} ts={p['ts']:.3f})  Δt={dt:.3f}s")
                for p in rd["fp"]:
                    print(f"[FP]  PRED(fid={p['fid']} ts={p['ts']:.3f})")
                for g in rd["fn"]:
                    print(f"[FN]  GT(fid={g['fid']} ts={g['ts']:.3f})")

    if not per_date_A:
        print(f"[WARN] {d} 沒有 Stage-A 結果")
        return
    
    # ─────────────── Micro 統計 ───────────────
    sumA = {k: sum(len(d[k]) for d in per_date_A)
            for k in ("full","part","fp","fn")}
    print("*** sumA ***")
    print(sumA)
    P_micro_A = safe_div(2*sumA["full"]+sumA["part"],
                         2*(sumA["full"]+sumA["part"]+sumA["fp"]))
    R_micro_A = safe_div(2*sumA["full"]+sumA["part"],
                         2*(sumA["full"]+sumA["part"]+sumA["fn"]))
    F1_micro_A = safe_div(2*P_micro_A*R_micro_A,
                          P_micro_A+R_micro_A) if (P_micro_A+R_micro_A) else 0

    sumB = {k: sum(d[k] for d in per_date_B)
            for k in ("tp_c","tp_sub","fp","fn")}
    print("*** sumB ***")
    print(sumB)
    eff_tp = sumB["tp_c"]
    P_micro_B = safe_div(eff_tp, eff_tp + sumB["fp"] + sumB["tp_sub"])
    R_micro_B = safe_div(eff_tp, eff_tp + sumB["fn"] + sumB["tp_sub"])
    F1_micro_B = safe_div(2*P_micro_B*R_micro_B,
                          P_micro_B+R_micro_B) if (P_micro_B+R_micro_B) else 0

    # ─────────────── Macro 統計 ───────────────
    print('***')
    print(per_rally_A_all)
    print('***')
    print(per_rally_B_all)
    P_macro_A,R_macro_A,F1_macro_A=_macro_from_rallies(per_rally_A_all)
    P_macro_B,R_macro_B,F1_macro_B=_macro_from_rallies(per_rally_B_all)

    # ─────────────── 列印總結 ───────────────
    n = len(per_rally_A_all)
    print("\n" + "="*72)
    print(f"  Evaluated {n} rally(s)  –  Micro vs. Macro Summary")
    print("="*72)
    print("Stage-A Serve/Dead")
    print(f"  Micro: P {pct(P_micro_A)}  R {pct(R_micro_A)}  F1 {pct(F1_micro_A)}")
    print(f"  Macro: P {pct(P_macro_A)}  R {pct(R_macro_A)}  F1 {pct(F1_macro_A)}")
    print("-"*72)
    print("Stage-B Hit")
    print(f"  Micro: P {pct(P_micro_B)}  R {pct(R_micro_B)}  F1 {pct(F1_micro_B)}")
    print(f"  Macro: P {pct(P_macro_B)}  R {pct(R_macro_B)}  F1 {pct(F1_macro_B)}")
    print("="*72)

if __name__=="__main__":
    main()
