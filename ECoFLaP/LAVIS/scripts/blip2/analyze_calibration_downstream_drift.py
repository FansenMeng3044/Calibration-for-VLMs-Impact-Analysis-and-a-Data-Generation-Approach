#!/usr/bin/env python3
"""Part 2 -- does downstream fidelity to dense predict accuracy across calibrations?

If the calibration main effect is NOT in the mask geometry (centrality failed),
the remaining place is downstream: how faithfully each pruned model reproduces
the dense model's behavior on eval data. This runs dense + the N joint
checkpoints on the SAME eval rows and, per checkpoint, measures distance to dense:

  - encoder rel-L2 drift, split by visual / text positions
  - decoder rel-L2 drift on the teacher-forced answer positions
  - answer-logit KL(dense || pruned)      <- closest thing to accuracy
  - answer top-1 agreement with dense

Then it correlates each per-checkpoint metric with the column-centered accuracy
main effect. The predictor that tracks accuracy is the mechanism: e.g. if
answer-logit KL is strongly NEGATIVELY correlated (less drift -> higher
accuracy) and OKVQA (the best, geometry-outlier calibration) has the LOWEST KL,
the resolution is "OKVQA's mask is a geometric outlier but functionally the
closest to dense."

Dense activations/logits are cached to disk per batch (float16 for storage;
KL is recomputed in float32 at compare time), so RAM stays bounded for any N.

Usage:
  python scripts/blip2/analyze_calibration_downstream_drift.py \
      --eval_json /p/mmbench_dev.json --images_dir /p/mmbench_images \
      --ckpt MMBench=/p/joint_mmbench.pth ... --ckpt cc3m=/p/joint_cc3m.pth \
      --accuracy_csv /p/accuracy_matrix.csv \
      --out_dir /p/out/part2_downstream --max_samples 64 --batch_size 2
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
from typing import Any, Dict, List, Optional

import numpy as np

from split_joint_analysis_common import (
    AUTO_INPUT_FIELDS, AUTO_OUTPUT_FIELDS, EncoderForward, build_vis_processor, ensure_dir,
    extract_text, iter_batches, load_batch_images, load_blip2, load_rows, parse_labeled_paths,
    select_rows, setup_matplotlib, write_csv,
)
from analyze_calibration_mask_mechanism import load_accuracy_effects

ENC_GROUPS = ("visual", "text")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-checkpoint downstream drift-to-dense vs accuracy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_json", required=True)
    p.add_argument("--images_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--dense_ckpt", default=None)
    p.add_argument("--ckpt", action="append", required=True, metavar="LABEL=PATH")
    p.add_argument("--accuracy_csv", default=None)
    p.add_argument("--model_name", default="blip2_t5")
    p.add_argument("--model_type", default="pretrain_flant5xl")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=64)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--image_size", type=int, default=224)
    p.add_argument("--image_field", default="image")
    p.add_argument("--text_field", default="auto")
    p.add_argument("--output_field", default="auto")
    p.add_argument("--max_txt_len", type=int, default=32)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--keep_cache", action="store_true", help="Do not delete the dense disk cache.")
    return p.parse_args()


def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    return (sums / counts)[inv]


def pearson(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    return float("nan") if x.std() == 0 or y.std() == 0 else float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(rankdata(np.asarray(x, float)), rankdata(np.asarray(y, float)))


class BlockCapture:
    def __init__(self, model, torch):
        self.torch = torch
        self.buffers: Dict[Any, Any] = {}
        self.handles = []
        for i, b in enumerate(model.t5_model.encoder.block):
            self.handles.append(b.register_forward_hook(self._h("enc", i)))
        for i, b in enumerate(model.t5_model.decoder.block):
            self.handles.append(b.register_forward_hook(self._h("dec", i)))

    def _h(self, part, i):
        def hook(_m, _in, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            self.buffers[(part, i)] = h.detach().to(self.torch.float32)
        return hook

    def clear(self): self.buffers = {}

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def run_forward(model, forward, args, batch_rows, start, original_indices, torch, Image, vis_processor,
                capture, want_logits):
    images = load_batch_images(batch_rows, start, original_indices, os.path.abspath(args.images_dir),
                               args.image_field, vis_processor, torch, Image).to(args.device)
    texts = [extract_text(r, args.text_field, AUTO_INPUT_FIELDS, original_indices[start + i])
             for i, r in enumerate(batch_rows)]
    capture.clear()
    out = forward.run(images, texts, args.device)
    enc_masks = {g: out["%s_mask" % g] for g in ENC_GROUPS}

    logits = None
    amask = None
    if want_logits:
        answers = [extract_text(r, args.output_field, AUTO_OUTPUT_FIELDS, original_indices[start + i])
                   for i, r in enumerate(batch_rows)]
        with torch.no_grad():
            with forward._amp(torch.bfloat16):
                tgt = model.t5_tokenizer(answers, padding="max_length", truncation=True,
                                         max_length=args.max_txt_len, return_tensors="pt").to(args.device)
                labels = tgt.input_ids.masked_fill(tgt.input_ids == model.t5_tokenizer.pad_token_id, -100)
                dec = model.t5_model(encoder_outputs=(out["encoder_hidden"],),
                                     attention_mask=out["encoder_attention"],
                                     decoder_attention_mask=tgt.attention_mask, labels=labels, return_dict=True)
                logits = dec.logits.to(torch.float32)
        amask = labels != -100
    return out, enc_masks, logits, amask


def main() -> None:
    args = parse_args()
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Missing runtime dependency: %s" % exc) from exc
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    ensure_dir(args.out_dir)
    cache_dir = os.path.join(args.out_dir, "_dense_cache")
    ensure_dir(cache_dir)

    pruned = parse_labeled_paths(args.ckpt)
    for lab, path in pruned.items():
        if not os.path.isfile(path):
            raise FileNotFoundError("Checkpoint not found for %s: %s" % (lab, path))

    rows, original_indices = select_rows(load_rows(os.path.abspath(args.eval_json)),
                                         args.max_samples, args.shuffle, args.seed)
    print("device:", args.device, "| samples:", len(rows), "| models: dense +", list(pruned))

    # ---------- dense pass: cache to disk ----------
    print("\n>>> dense pass (caching to disk)")
    model = load_blip2(args.model_name, args.model_type, args.device, args.dense_ckpt, args.max_txt_len)
    if args.fp32:
        model.float()
    capture = BlockCapture(model, torch)
    forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
    vis_processor = build_vis_processor(args.image_size)
    n_batches = 0
    try:
        for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
            out, enc_masks, logits, amask = run_forward(
                model, forward, args, batch_rows, start, original_indices, torch, Image, vis_processor,
                capture, want_logits=True)
            # Block activations MUST be float32: T5's text-position activation outliers
            # exceed fp16's 65504 ceiling, turning the cache into inf and every text-token
            # rel-L2 into nan. Logits (~+-30) are safe in fp16.
            save = {"vmask": enc_masks["visual"].cpu().numpy(), "tmask": enc_masks["text"].cpu().numpy(),
                    "amask": amask.cpu().numpy(), "logits": logits.to(torch.float16).cpu().numpy()}
            for (part, i), h in capture.buffers.items():
                save["blk__%s__%d" % (part, i)] = h.to(torch.float32).cpu().numpy()
            np.savez(os.path.join(cache_dir, "batch_%d.npz" % bi), **save)
            n_batches += 1
    finally:
        capture.close(); del model; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- pruned passes ----------
    def compare_pass(label, path):
        print("\n>>> [%s] %s" % (label, path))
        model = load_blip2(args.model_name, args.model_type, args.device, path, args.max_txt_len)
        if args.fp32:
            model.float()
        capture = BlockCapture(model, torch)
        forward = EncoderForward(model, torch, padding="max_length", fp32=args.fp32)
        acc = {("enc", "visual"): [], ("enc", "text"): [], ("dec", "answer"): []}
        kl_vals: List[float] = []
        top1_hits = 0
        top1_tot = 0
        try:
            for bi, (start, batch_rows) in enumerate(iter_batches(rows, args.batch_size)):
                dense = np.load(os.path.join(cache_dir, "batch_%d.npz" % bi))
                out, enc_masks, logits, amask = run_forward(
                    model, forward, args, batch_rows, start, original_indices, torch, Image, vis_processor,
                    capture, want_logits=True)
                # block drift
                for (part, i), h in capture.buffers.items():
                    key = "blk__%s__%d" % (part, i)
                    if key not in dense.files:
                        continue
                    ref = torch.from_numpy(dense[key]).to(h.device, torch.float32)
                    if ref.shape != h.shape:
                        raise SystemExit("[FATAL] shape mismatch %s block %d" % (part, i))
                    rel = (torch.linalg.norm(h - ref, dim=-1) /
                           torch.linalg.norm(ref, dim=-1).clamp_min(1e-6))
                    if part == "enc":
                        for g in ENC_GROUPS:
                            mask_key = "vmask" if g == "visual" else "tmask"
                            m = torch.from_numpy(dense[mask_key]).to(h.device)
                            if bool(m.any()):
                                vals = rel[m].detach().cpu().numpy()
                                if not np.all(np.isfinite(vals)):
                                    raise SystemExit(
                                        "[FATAL] non-finite rel-L2 at enc/%s block %d -- numerics bug "
                                        "(fp16 overflow?), do NOT interpret." % (g, i))
                                acc[("enc", g)].extend(vals.tolist())
                    else:
                        am = torch.from_numpy(dense["amask"]).to(h.device)
                        if bool(am.any()):
                            vals = rel[am].detach().cpu().numpy()
                            if not np.all(np.isfinite(vals)):
                                raise SystemExit(
                                    "[FATAL] non-finite rel-L2 at dec block %d -- numerics bug." % i)
                            acc[("dec", "answer")].extend(vals.tolist())
                # answer-logit KL + top1 (fp32 recompute)
                dl = torch.from_numpy(dense["logits"]).to(logits.device, torch.float32)
                am = torch.from_numpy(dense["amask"]).to(logits.device)
                if bool(am.any()):
                    lp = torch.log_softmax(dl, dim=-1)
                    lq = torch.log_softmax(logits, dim=-1)
                    kl = (lp.exp() * (lp - lq)).sum(dim=-1)
                    kl_vals.extend(kl[am].detach().cpu().numpy().tolist())
                    top1_hits += int((dl.argmax(-1)[am] == logits.argmax(-1)[am]).sum().item())
                    top1_tot += int(am.sum().item())
        finally:
            capture.close(); del model; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return {
            "model": label,
            "enc_visual_rel_l2": float(np.mean(acc[("enc", "visual")])) if acc[("enc", "visual")] else float("nan"),
            "enc_text_rel_l2": float(np.mean(acc[("enc", "text")])) if acc[("enc", "text")] else float("nan"),
            "dec_answer_rel_l2": float(np.mean(acc[("dec", "answer")])) if acc[("dec", "answer")] else float("nan"),
            "answer_logit_kl": float(np.mean(kl_vals)) if kl_vals else float("nan"),
            "answer_top1_agree": (top1_hits / top1_tot) if top1_tot else float("nan"),
            "answer_tokens": top1_tot,
        }

    per_ckpt = [compare_pass(lab, path) for lab, path in pruned.items()]
    write_csv(os.path.join(args.out_dir, "downstream_drift_per_checkpoint.csv"), per_ckpt)

    if not args.keep_cache:
        shutil.rmtree(cache_dir, ignore_errors=True)

    # ---------- correlate with accuracy ----------
    corr_rows = []
    effects = None
    if args.accuracy_csv:
        effects, _, _, _ = load_accuracy_effects(args.accuracy_csv)
        metrics = ["enc_visual_rel_l2", "enc_text_rel_l2", "dec_answer_rel_l2",
                   "answer_logit_kl", "answer_top1_agree"]
        shared = [r for r in per_ckpt if r["model"] in effects]
        y = np.asarray([effects[r["model"]] for r in shared], float)
        for m in metrics:
            xs = np.asarray([r[m] for r in shared], float)
            ok = np.isfinite(xs)
            if ok.sum() >= 3 and np.nanstd(xs[ok]) > 0:
                corr_rows.append({"metric": m, "n": int(ok.sum()),
                                  "pearson_r": pearson(xs[ok], y[ok]),
                                  "spearman_rho": spearman(xs[ok], y[ok])})
        corr_rows.sort(key=lambda r: abs(r["spearman_rho"]) if np.isfinite(r["spearman_rho"]) else 0, reverse=True)
        write_csv(os.path.join(args.out_dir, "drift_accuracy_correlations.csv"), corr_rows)

    plt = setup_matplotlib()
    if plt is not None and effects is not None:
        shared = [r for r in per_ckpt if r["model"] in effects]
        xs = [r["answer_logit_kl"] for r in shared]; ys = [effects[r["model"]] for r in shared]
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(xs, ys, s=140, color="#4C78A8", edgecolors="black", zorder=3)
        for r in shared:
            ax.annotate(r["model"], (r["answer_logit_kl"], effects[r["model"]]), fontsize=11,
                        xytext=(7, 5), textcoords="offset points")
        ax.set_xlabel("answer-logit KL to dense  (lower = more faithful)")
        ax.set_ylabel("global accuracy effect")
        ax.set_title("Does staying close to dense predict accuracy?")
        ax.grid(True, alpha=0.28); fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "kl_vs_accuracy.png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as h:
        json.dump({"per_checkpoint": per_ckpt, "correlations": corr_rows,
                   "accuracy_effects": effects}, h, indent=2)

    print("\n=== per-checkpoint drift to dense ===")
    print("  %-10s %10s %10s %10s %9s %9s" % ("model", "enc_vis", "enc_text", "dec_ans", "logitKL", "top1"))
    for r in per_ckpt:
        print("  %-10s %10.4f %10.4f %10.4f %9.4f %9.3f" % (
            r["model"], r["enc_visual_rel_l2"], r["enc_text_rel_l2"], r["dec_answer_rel_l2"],
            r["answer_logit_kl"], r["answer_top1_agree"]))

    if corr_rows:
        print("\n=== correlation of drift metrics with accuracy effect ===")
        for r in corr_rows:
            print("  %-20s pearson=%+.3f  spearman=%+.3f" % (r["metric"], r["pearson_r"], r["spearman_rho"]))
        top = corr_rows[0]
        print("\n=== verdict ===")
        if abs(top["spearman_rho"]) >= 0.8:
            direction = "less drift -> higher accuracy" if top["spearman_rho"] < 0 else "more of it -> higher accuracy"
            print("  '%s' tracks accuracy (spearman %+.2f; %s)." % (top["metric"], top["spearman_rho"], direction))
            if top["metric"] == "answer_logit_kl" and top["spearman_rho"] < 0:
                print("  Mechanism found downstream: the best calibration is the one whose pruned model")
                print("  stays functionally closest to dense on the answer distribution -- regardless of")
                print("  where its mask sits geometrically. Central mask != faithful output.")
        else:
            print("  No downstream metric explains accuracy either (top |spearman|=%.2f)." % abs(top["spearman_rho"]))
            print("  At n=%d this may just be underpowered. Consider more eval samples, or that the" % top["n"])
            print("  accuracy differences are near run-to-run noise -- re-check the accuracy table itself.")

    print("\n[OK] wrote:", os.path.abspath(args.out_dir))


if __name__ == "__main__":
    main()
