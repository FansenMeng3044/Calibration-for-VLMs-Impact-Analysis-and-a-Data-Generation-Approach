#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证项 3：对照等价性。比较两个 checkpoint（如 ATV vs naive Wanda，均 CC3M / 只剪 T5 / 均匀）
在 T5 权重上的剪枝掩码差异——若 ATV 的 k 恒等于 num_img（退化），两者应逐权重相同。

用法:
  python scripts/blip2/compare_ckpts.py --a atv.pth --b wanda.pth
  python scripts/blip2/compare_ckpts.py --a atv.pth --b wanda.pth --prefix t5_model.encoder
"""
import argparse
import torch


def unwrap(obj):
    if isinstance(obj, dict):
        for k in ("model", "state_dict", "module"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k]
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--prefix", default="t5_model", help="只比较该前缀下的权重（默认整个 T5）")
    ap.add_argument("--min_numel", type=int, default=4096)
    args = ap.parse_args()

    sa = unwrap(torch.load(args.a, map_location="cpu"))
    sb = unwrap(torch.load(args.b, map_location="cpu"))

    keys = [k for k in sa
            if k in sb and torch.is_tensor(sa[k]) and sa[k].dim() >= 2
            and sa[k].numel() >= args.min_numel and k.startswith(args.prefix)]
    if not keys:
        raise SystemExit(f"[FATAL] 没有可比较的权重（prefix={args.prefix}）")

    tot = 0
    mask_diff = 0        # 剪枝掩码（zero-status）不同的位置数
    identical_layers = 0
    diff_layers = 0
    max_abs = 0.0
    for k in keys:
        wa, wb = sa[k].float(), sb[k].float()
        if wa.shape != wb.shape:
            print(f"  [SKIP] {k}: shape {tuple(wa.shape)} vs {tuple(wb.shape)}")
            continue
        za, zb = (wa == 0), (wb == 0)
        d = int((za ^ zb).sum().item())     # 掩码异或
        n = int(wa.numel())
        tot += n
        mask_diff += d
        wl = (wa - wb).abs().max().item()
        max_abs = max(max_abs, wl)
        if d == 0 and wl == 0.0:
            identical_layers += 1
        else:
            diff_layers += 1

    print(f"\n===== 比较 =====\n  A: {args.a}\n  B: {args.b}\n  prefix: {args.prefix}")
    print(f"  权重张量: {len(keys)}  (逐权重相同 {identical_layers} / 不同 {diff_layers})")
    print(f"  剪枝掩码不同位置: {mask_diff:,} / {tot:,} = {100.0 * mask_diff / max(1, tot):.4f}%")
    print(f"  最大逐元素权重差: {max_abs:.3e}")
    if mask_diff == 0 and max_abs == 0.0:
        print("  结论: 两 ckpt 逐权重完全相同 → ATV 退化为 naive Wanda（k 恒=num_img）。")
    else:
        print("  结论: 两 ckpt 不同 → ATV 的 token 选择产生了不同的剪枝掩码（非退化）。")


if __name__ == "__main__":
    main()
