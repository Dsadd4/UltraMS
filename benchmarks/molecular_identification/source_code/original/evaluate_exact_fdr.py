#!/usr/bin/env python3
"""Build tie-complete exact FDR frontiers for all Figure 3h methods."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import shutil
from pathlib import Path

import numpy as np

from baseline_common import atomic_write_json, exact_tie_aware_fdr_curve, fdr_working_point, sha256_file


def exact_low_fdr_average(curve:dict[str,np.ndarray],target:float)->dict:
    fdr=np.asarray(curve["fdr"],float); recall=np.asarray(curve["recall"],float)
    valid=np.isfinite(fdr)&(fdr>=0)&(fdr<=target)
    breaks=np.unique(np.r_[0.0,fdr[valid],target]); breaks.sort(); area=0.0
    for left,right in zip(breaks[:-1],breaks[1:]):
        eligible=np.isfinite(fdr)&(fdr<=left)
        value=float(recall[eligible].max()) if eligible.any() else 0.0
        area+=(right-left)*value
    return {"fdr_max":target,"auc":area,"average_recall":area/target if target>0 else 0.0,
            "integration":"exact step envelope over observed empirical-FDR breakpoints"}


def load_existing(path:Path)->list[dict]:
    summary=json.loads(path.read_text()); records=[]
    for mode in ("pos","neg"):
        for model,label in (("ultra","UltraMS"),("dreams","DreaMS")):
            dist=summary["adducts"][mode]["models"][model]["score_dist"]
            records.append({"method":label,"mode":mode,"split":"test","seed":None,
                            "score":np.asarray(dist["top1_sim"],np.float32),
                            "correct":np.asarray(dist["top1_correct"],bool),"has_positive":None,
                            "source":str(path),"source_sha256":sha256_file(path)})
    return records


def load_npz(specification:str)->dict:
    parts=specification.split("|",4)
    if len(parts)!=5: raise ValueError("--input must be method|mode|split|seed|path")
    method,mode,split,seed_text,path_text=parts; path=Path(path_text); data=np.load(path,allow_pickle=False)
    score=data["top1_score"] if "top1_score" in data else data["top1_sims"]
    correct=data["top1_correct"] if "top1_correct" in data else data["correct"]
    has=data["has_library_positive"] if "has_library_positive" in data else None
    return {"method":method,"mode":mode,"split":split,"seed":None if seed_text in ("","none") else int(seed_text),
            "score":np.asarray(score,np.float32),"correct":np.asarray(correct,bool),
            "has_positive":None if has is None else np.asarray(has,bool),"source":str(path),"source_sha256":sha256_file(path)}


def run(args:argparse.Namespace)->None:
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(Path(args.manifest_summary).read_text()); records=[]
    if args.existing_summary: records.extend(load_existing(Path(args.existing_summary)))
    for specification in args.input: records.append(load_npz(specification))
    manifest_copy=output/"strict_adduct_manifest_summary.json"
    shutil.copy2(args.manifest_summary,manifest_copy)
    atomic_write_json(output/"config.json",{
        "protocol":"exact tie-complete observed-score FDR frontier",
        "target_fdr":args.target_fdr,"minimum_accepted":args.min_accepted,
        "strict_adduct_manifest_sha256":sha256_file(manifest_copy),
        "inputs":[{"method":record["method"],"mode":record["mode"],"split":record["split"],
                   "seed":record["seed"],"source_sha256":record["source_sha256"]} for record in records]})
    summaries=[]; frontier_path=output/"exact_fdr_frontiers.csv.gz"
    with gzip.open(frontier_path,"wt",newline="") as handle:
        fields=["method","mode","adduct","split","seed","threshold","accepted","errors","correct",
                "empirical_fdr","identification_recall","correct_hit_retention","n_eligible_queries","n_unthresholded_correct"]
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for record in records:
            mode=record["mode"]; split=record["split"]
            if record["has_positive"] is not None: denominator=int(record["has_positive"].sum())
            elif split=="test": denominator=int(manifest["modes"][mode]["test"]["n_queries_with_library_positive"])
            else: denominator=int(manifest["modes"][mode]["validation"]["n_queries_with_library_positive"])
            curve=exact_tie_aware_fdr_curve(record["score"],record["correct"],denominator=denominator,min_hits=args.min_accepted)
            n_correct_total=int(record["correct"].sum()); work=fdr_working_point(curve,args.target_fdr)
            low=exact_low_fdr_average(curve,args.target_fdr)
            for threshold,accepted,correct,fdr,recall in zip(curve["threshold"],curve["accepted"],curve["correct"],curve["fdr"],curve["recall"]):
                writer.writerow({"method":record["method"],"mode":mode,"adduct":"[M+H]+" if mode=="pos" else "[M-H]-",
                                 "split":split,"seed":"" if record["seed"] is None else record["seed"],
                                 "threshold":float(threshold),"accepted":int(accepted),"errors":int(accepted-correct),"correct":int(correct),
                                 "empirical_fdr":"" if not np.isfinite(fdr) else float(fdr),"identification_recall":float(recall),
                                 "correct_hit_retention":float(correct/n_correct_total) if n_correct_total else 0.0,
                                 "n_eligible_queries":denominator,"n_unthresholded_correct":n_correct_total})
            standardized=output/f"top1_{record['method'].lower().replace(' ','_').replace('+','plus')}_{mode}_{split}_seed_{record['seed'] if record['seed'] is not None else 'na'}.npz"
            standardized_arrays={"top1_score":record["score"],"top1_correct":record["correct"]}
            if record["has_positive"] is not None:
                standardized_arrays["has_library_positive"]=record["has_positive"]
            np.savez_compressed(standardized,**standardized_arrays)
            summaries.append({"method":record["method"],"mode":mode,"split":split,"seed":record["seed"],
                              "n_query":len(record["score"]),"n_eligible_queries":denominator,
                              "n_unthresholded_correct":n_correct_total,"unthresholded_hit_at_1":float(record["correct"].mean()),
                              "working_point":work,"low_fdr":low,"n_exact_frontier_points":len(curve["threshold"]),
                              "source":record["source"],"source_sha256":record["source_sha256"],
                              "standardized_top1_sha256":sha256_file(standardized)})
    done={"status":"complete","target_fdr":args.target_fdr,"minimum_accepted":args.min_accepted,
          "manifest_summary_sha256":sha256_file(args.manifest_summary),"frontier_sha256":sha256_file(frontier_path),
          "manifest_copy_sha256":sha256_file(manifest_copy),"config_sha256":sha256_file(output/"config.json"),
          "summaries":summaries}
    atomic_write_json(output/"exact_fdr_summary.json",done); print(json.dumps(done,indent=2),flush=True)


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--manifest-summary",required=True); p.add_argument("--existing-summary")
    p.add_argument("--input",action="append",default=[]); p.add_argument("--output-dir",required=True)
    p.add_argument("--target-fdr",type=float,default=.05); p.add_argument("--min-accepted",type=int,default=10); run(p.parse_args())


if __name__=="__main__": main()
