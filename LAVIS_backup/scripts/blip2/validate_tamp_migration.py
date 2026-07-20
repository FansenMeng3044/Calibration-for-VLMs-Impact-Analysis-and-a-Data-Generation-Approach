#!/usr/bin/env python3
"""Static validation for the BLIP2-T5 TAMP migration.

This script is intentionally dependency-light: it checks source-level invariants
that protect the AMIA/DAS migration fixes without loading BLIP2-T5 or torch.
Use it before the heavier GPU smoke run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def find_between(text: str, start: str, end: str) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    end_idx = text.find(end, start_idx + len(start))
    if end_idx < 0:
        return text[start_idx:]
    return text[start_idx:end_idx]


def add_check(rows: List[Dict[str, object]], name: str, passed: bool, evidence: str) -> None:
    rows.append({"check": name, "passed": bool(passed), "evidence": evidence})


def validate(lavis_root: Path) -> Tuple[bool, List[Dict[str, object]]]:
    rows: List[Dict[str, object]] = []

    wanda_path = lavis_root / "lavis" / "compression" / "pruners" / "wanda_pruner.py"
    layer_path = lavis_root / "lavis" / "compression" / "pruners" / "layer_single_base_pruner.py"
    eval_path = lavis_root / "evaluate_blip.py"
    unimodal_path = lavis_root / "lavis" / "compression" / "unimodal_prune.py"
    blip2_t5_path = lavis_root / "lavis" / "models" / "blip2_models" / "blip2_t5.py"
    modeling_t5_path = lavis_root / "lavis" / "models" / "blip2_models" / "modeling_t5.py"
    smoke_path = lavis_root / "scripts" / "blip2" / "smoke_tamp_migration_runtime.py"
    core_smoke_path = lavis_root / "scripts" / "blip2" / "smoke_tamp_core_ops.py"
    validation_runner_path = lavis_root / "scripts" / "blip2" / "run_tamp_migration_validation.sh"
    validation_checker_path = lavis_root / "scripts" / "blip2" / "check_tamp_validation_outputs.py"
    validation_doc_path = lavis_root / "scripts" / "blip2" / "TAMP_MIGRATION_VALIDATION.md"
    c4_text_runner_path = (
        lavis_root / "scripts" / "blip2" / "run_c4_tamp_llmonly_prune_then_dual_fourbench_eval.sh"
    )
    c4_prune_runner_path = (
        lavis_root / "scripts" / "blip2" / "run_lavisbackup_prune_t5_c4_llm_only.sh"
    )
    six_text_runner_path = (
        lavis_root / "scripts" / "blip2" / "run_lavisbackup_tamp_textcalib_six_prune_eval.sh"
    )

    files = {
        "wanda": wanda_path,
        "layer": layer_path,
        "evaluate": eval_path,
        "unimodal": unimodal_path,
        "blip2_t5": blip2_t5_path,
        "modeling_t5": modeling_t5_path,
        "runtime_smoke": smoke_path,
        "core_smoke": core_smoke_path,
        "validation_runner": validation_runner_path,
        "validation_checker": validation_checker_path,
        "validation_doc": validation_doc_path,
        "c4_text_runner": c4_text_runner_path,
        "c4_prune_runner": c4_prune_runner_path,
        "six_text_runner": six_text_runner_path,
    }
    for label, path in files.items():
        add_check(rows, f"{label} source exists", path.is_file(), str(path))

    if not all(path.is_file() for path in files.values()):
        return False, rows

    wanda = read_text(wanda_path)
    layer = read_text(layer_path)
    evaluate = read_text(eval_path)
    unimodal = read_text(unimodal_path)
    blip2_t5 = read_text(blip2_t5_path)
    modeling_t5 = read_text(modeling_t5_path)
    smoke = read_text(smoke_path)
    core_smoke = read_text(core_smoke_path)
    validation_runner = read_text(validation_runner_path)
    validation_checker = read_text(validation_checker_path)
    validation_doc = read_text(validation_doc_path)
    c4_text_runner = read_text(c4_text_runner_path)
    c4_prune_runner = read_text(c4_prune_runner_path)
    six_text_runner = read_text(six_text_runner_path)
    text_only_runners = "\n".join([c4_text_runner, c4_prune_runner, six_text_runner])

    t5_section = find_between(
        wanda,
        '@registry.register_pruner("t5_wanda_pruner")',
        '@registry.register_pruner("vit_wanda_pruner")',
    )
    vit_section = find_between(
        wanda,
        '@registry.register_pruner("vit_wanda_pruner")',
        '@registry.register_pruner("blipt5_wanda_pruner")',
    )
    blipt5_section = find_between(
        wanda,
        '@registry.register_pruner("blipt5_wanda_pruner")',
        "def run_pruner",
    )
    t5_prepare = find_between(
        t5_section,
        "    def prepare_calibration_input_encoder(",
        "    @print_time\n    def _prune(",
    )
    t5_prune = find_between(
        t5_section,
        "    def _prune(",
        "    def get_sparsity(",
    )
    vit_prepare = find_between(
        vit_section,
        "    def prepare_calibration_input_encoder(",
        "    @print_time\n    def _prune(",
    )
    amia_select = find_between(
        wanda,
        "    def _select_tokens(self, out, image_mask, score, attention_mask=None, eps=1e-8):",
        "    def add_batch(self, inp, out, image_mask=None, score=None, attention_mask=None):",
    )

    add_check(
        rows,
        "TAMP alias derives max_sparsity_per_layer from sparsity + 0.1",
        'args.pruning_method == "blipt5_tamp_pruner"' in evaluate
        and "tamp_sparsity = 1.0 - keep_ratio" in evaluate
        and "args.max_sparsity_per_layer = min(1.0, tamp_sparsity + 0.1)" in evaluate,
        "evaluate_blip.py blipt5_tamp_pruner alias",
    )
    add_check(
        rows,
        "TAMP alias refuses to prune without a prune spec",
        "will_run_pruner = args.save_pruned_model" in evaluate
        and "prune_spec is None and will_run_pruner" in evaluate
        and "unsafe default 0.8" in evaluate,
        "evaluate_blip.py blipt5_tamp_pruner missing-spec guard",
    )
    add_check(
        rows,
        "TAMP CLI help documents multimodal AMIA and pure-text Wanda degeneration",
        "multimodal uses amia + density_sum + layer" in evaluate
        and "pure text calibration degrades to vanilla Wanda" in evaluate,
        "evaluate_blip.py --pruning_method help text",
    )
    add_check(
        rows,
        "TAMP alias runs the single-modality reduction for text-only calibration",
        'if args.prune_calib_mode == "t5_c4_text":' in evaluate
        and "single-modality reduction" in evaluate
        # The vanilla-Wanda degradation branch was removed on purpose; the naive+uniform
        # baseline now lives under --pruning_method blipt5_wanda_pruner instead.
        and 'args.token_selection = "naive"' not in evaluate,
        "evaluate_blip.py blipt5_tamp_pruner text-only branch",
    )
    add_check(
        rows,
        "Pure-text TAMP scripts prune only T5 and rely on Wanda degeneration",
        text_only_runners.count("--prune_calib_mode t5_c4_text") >= 3
        and text_only_runners.count("--t5_prune_spec") >= 3
        and "--prune_vit" not in text_only_runners
        and "--no_prune_t5" not in text_only_runners
        and "blipt5_tamp_pruner" in text_only_runners,
        "text-only TAMP runner scripts",
    )
    add_check(
        rows,
        "evaluate_blip defaults to T5-only pruning unless ViT is explicitly requested",
        'parser.add_argument(\n        "--prune_vit"' in evaluate
        and "Default is already to skip ViT" in evaluate
        and "elif args.prune_vit:" in evaluate
        and "args.no_prune_vit = True" in evaluate,
        "evaluate_blip.py prune_vit/no_prune_vit defaults",
    )
    add_check(
        rows,
        "TAMP alias uses AMIA plus DAS only for multimodal calibration",
        'args.token_selection = "amia"' in evaluate
        and 'args.score_method = "density_sum"' in evaluate
        and 'args.sparsity_ratio_granularity = "layer"' in evaluate
        and evaluate.find('args.token_selection = "amia"')
        > evaluate.find('if args.prune_calib_mode == "t5_c4_text":'),
        "evaluate_blip.py blipt5_tamp_pruner multimodal branch",
    )
    add_check(
        rows,
        "TAMP multimodal alias rejects ViT-only configurations",
        "using_tamp_alias = args.pruning_method == \"blipt5_tamp_pruner\"" in evaluate
        and "blipt5_tamp_pruner multimodal mode requires T5 pruning" in evaluate
        and "importance_scope={args.importance_scope}" in evaluate
        and "AMIA/DAS require T5 encoder tokens" in evaluate,
        "evaluate_blip.py blipt5_tamp_pruner invalid ViT-only guard",
    )
    add_check(
        rows,
        "AMIA derives column-wise encoder attention scores",
        "def _encoder_attention_column_scores" in wanda
        and "rows = head_mean[b][valid_b]" in wanda
        and "score_b = rows.mean(dim=0)" in wanda
        and "score_b.masked_fill(~valid_b, 0.0)" in wanda,
        "wanda_pruner.py _encoder_attention_column_scores",
    )
    add_check(
        rows,
        "BLIP2-T5 forward exposes visual/text token labels and encoder attention masks",
        "encoder_atts = torch.cat([atts_t5, input_tokens.attention_mask], dim=1)" in blip2_t5
        and "temp_label[:, :num_query] = True" in blip2_t5
        and "self.temp_label = temp_label" in blip2_t5
        and "self.temp_encoder_atts = encoder_atts.detach()" in blip2_t5,
        "blip2_t5.py forward temp_label/temp_encoder_atts",
    )
    add_check(
        rows,
        "T5Block output order matches helper assumptions",
        "outputs = (hidden_states,)" in modeling_t5
        and "outputs = outputs + (present_key_value_state,) + attention_outputs" in modeling_t5
        and "outputs = outputs + attention_outputs" in modeling_t5
        and "return outputs  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)" in modeling_t5,
        "modeling_t5.py T5Block.forward tuple order",
    )
    add_check(
        rows,
        "AMIA no longer silently falls back to all-one contribution scores",
        "AMIA token selection requires attention contribution scores" in wanda
        and "score = torch.ones(N" not in wanda,
        "AdaptiveMultimodalInputActivation._select_tokens",
    )
    add_check(
        rows,
        "AMIA scores are produced inside the T5 pruning loop",
        "scores_this_layer = [None] * num_cached_batches" in wanda
        and "_normal_t5_block_forward(" in wanda
        and "output_attentions=True" in wanda
        and "scores_this_layer[j] = _encoder_attention_column_scores" in wanda,
        "T5LayerWandaPruner._prune",
    )
    add_check(
        rows,
        "AMIA token selection is per-sample and padding-aware",
        "def add_batch(self, inp, out, image_mask=None, score=None, attention_mask=None)" in wanda
        and "for b in range(B):" in wanda
        and "valid_idx = torch.where(valid_b)[0]" in wanda
        and "attention_mask[b] if attention_mask is not None else None" in wanda,
        "AdaptiveMultimodalInputActivation.add_batch",
    )
    add_check(
        rows,
        "T5 pruning loop distinguishes cached batches from physical samples",
        "num_cached_batches = min(n_samples, len(inps))" in t5_section
        and "for j in range(num_cached_batches):" in t5_section
        and "processed_physical_samples = sample_offset" in t5_section
        and "nsamples == processed_physical_samples" in t5_section
        and "len(inps) * inps[0].shape[0]" not in t5_section,
        "T5LayerWandaPruner._prune cached batch accounting",
    )
    add_check(
        rows,
        "AMIA selection loop prevents reselection and updates selected-token floor after neighbor penalty",
        "available = torch.ones(N, dtype=torch.bool, device=out.device)" in amia_select
        and "masked_score = graph_score.masked_fill(~available, -torch.inf)" in amia_select
        and amia_select.find("for nb in neighbors:") >= 0
        and amia_select.find("min_val = graph_score.min().item() - 1.0")
        > amia_select.find("for nb in neighbors:"),
        "AdaptiveMultimodalInputActivation._select_tokens loop",
    )
    add_check(
        rows,
        "AMIA density term is computed once outside the selection loop",
        amia_select.find("density = _cos_pairwise_density_single") >= 0
        and amia_select.find("while True:") > amia_select.find("density = _cos_pairwise_density_single"),
        "AdaptiveMultimodalInputActivation._select_tokens density",
    )
    add_check(
        rows,
        "DAS cosine density excludes padding tokens",
        "def cos_pairwise_density(embeddings, image_mask, attention_mask=None" in layer
        and "mask = mask & valid" in layer
        and "l_idx = torch.where((~mask) & valid)[0]" in layer
        and "fill_value=True" not in layer,
        "layer_single_base_pruner.py cos_pairwise_density",
    )
    add_check(
        rows,
        "DAS requires raw encoder attention masks before density computation",
        "compute_density requires calibration_fn to return raw encoder_attention_masks" in layer
        and "compute_density calibration batch mismatch" in layer
        and "encoder_attention_masks[{idx}].shape" in layer,
        "LayerSparsity.compute_density mask contract",
    )
    add_check(
        rows,
        "DAS is computed per Linear rather than per T5 block output",
        "def find_layers(module, layers=(nn.Linear,), name=\"\")" in layer
        and "subset = find_layers(layer)" in layer
        and "full_name = f\"{param_encoder_prefix}{i}.{name}.weight\"" in layer
        and "wrapped_layers[name] = ActivationDensity()" in layer,
        "LayerSparsity.compute_density",
    )
    add_check(
        rows,
        "DAS derives the T5 encoder module path from the parameter prefix",
        "param_encoder_prefix = encoder_prefixes[0] if encoder_prefixes else default_t5_encoder_prefix" in layer
        and "t5_encoder_module_path = param_encoder_prefix.rstrip(\".\")" in layer
        and "_get_module_by_path(self.model, t5_encoder_module_path)" in layer,
        "LayerSparsity.compute_density module path inference",
    )
    add_check(
        rows,
        "DAS raises if requested encoder Linear layers are not observed",
        "matched_names = set()" in layer
        and "missing_names = sorted(set(layer_to_group_mapping) - matched_names)" in layer
        and "compute_density did not observe all requested T5 encoder Linear layers" in layer,
        "LayerSparsity.compute_density matched_names guard",
    )
    add_check(
        rows,
        "Density sparsity allocation only uses T5 encoder keys",
        "encoder_prefix = f\"{self.t5_model_prefix}.encoder.block.\"" in wanda
        and "parameters_for_allocation = [" in wanda
        and "k for k in parameters_to_prune if k.startswith(encoder_prefix)" in wanda,
        "BLIPT5LayerWandaPruner.get_sparsity",
    )
    add_check(
        rows,
        "Decoder and non-encoder tensors fall back to uniform sparsity under DAS",
        "sparsity.setdefault(name, original_sparsity)" in wanda,
        "get_sparsity density fallback",
    )
    add_check(
        rows,
        # Deliberately asserts the ECoFLaP convention, NOT propagation. Every block
        # replays with the arguments captured at block 0, so position_bias stays None
        # and blocks >0 fall back to a zero bias. Kept for comparability with the
        # ECoFLaP / Wanda / SparseGPT numbers this codebase reproduces.
        # NOTE: substring checks alone cannot see this -- the helper still *computes*
        # next_cache, it just has no caller. So assert the absence of the rebind.
        "T5 block replay reuses the block-0 cache (ECoFLaP convention, no propagation)",
        "def _normal_t5_block_forward" in wanda
        and "layer_caches = [dict(cache) for cache in caches]" in wanda
        and "_normal_t5_block_forward(layer, inps[j], layer_caches[j])" in wanda
        and "layer_caches = next_layer_caches" not in wanda
        and "next_layer_caches" not in wanda,
        "wanda_pruner.py T5 replay",
    )
    add_check(
        rows,
        "DAS T5 block replay helper matches decoder-aware T5Block tuple order",
        "def _normal_t5_block_forward(layer, hidden_states, cache, output_attentions=False)" in layer
        and "kwargs[\"output_attentions\"] = True" in layer
        and "cross_bias_index = bias_offset + 2 if output_attentions else bias_offset + 1" in layer
        and "next_cache[\"encoder_decoder_position_bias\"] = outputs[cross_bias_index].detach()" in layer,
        "layer_single_base_pruner.py _normal_t5_block_forward",
    )
    add_check(
        rows,
        "T5 pruning sections do not use stale direct layer(inps[j], **caches[j]) replay",
        "layer(inps[j], **caches[j])" not in t5_section
        and "layer(inps[j], **caches[j])" not in blipt5_section,
        "T5/BLIPT5 wanda sections",
    )
    add_check(
        rows,
        "Multimodal calibration requires temp_encoder_atts with matching shape",
        "model.temp_encoder_atts not found" in wanda
        and "requires encoder_attention_masks" in wanda
        and "encoder_attention_masks vs inps length mismatch" in wanda
        and "encoder_attention_masks[i].shape == (B, S)" in wanda,
        "prepare_calibration_input_encoder(return_image_masks=True)",
    )
    add_check(
        rows,
        "Calibration catchers use a private control-flow exception",
        "class _CatcherExit(Exception)" in wanda
        and "raise _CatcherExit" in wanda
        and "except _CatcherExit" in wanda
        and "except ValueError" not in t5_prepare,
        "wanda_pruner.py calibration catcher exception",
    )
    add_check(
        rows,
        "Calibration catchers restore wrapped layers and T5 use_cache in finally",
        "original_layer0 = layers[0]" in t5_prepare
        and "layers[0] = Catcher(original_layer0)" in t5_prepare
        and "finally:" in t5_prepare
        and "layers[0] = original_layer0" in t5_prepare
        and "config.use_cache = use_cache" in t5_prepare
        and "cache[k] = kwargs.get(k, None)" in t5_prepare
        and "original_layer0 = layers[0]" in vit_prepare
        and "layers[0] = Catcher(original_layer0)" in vit_prepare
        and "finally:" in vit_prepare
        and "layers[0] = original_layer0" in vit_prepare,
        "wanda_pruner.py prepare_calibration_input_encoder cleanup",
    )
    add_check(
        rows,
        "T5 pruning does not redundantly mutate global use_cache outside calibration capture",
        "config.use_cache = False" not in t5_prune
        and "config.use_cache = use_cache" not in t5_prune,
        "T5LayerWandaPruner._prune use_cache side effects",
    )
    add_check(
        rows,
        "Forward hooks are removed in finally during DAS and Wanda calibration forwards",
        "finally:" in t5_prune
        and "for h in handles:" in t5_prune
        and "h.remove()" in t5_prune
        and "finally:" in layer
        and "for h in handles:" in layer
        and "h.remove()" in layer,
        "wanda_pruner.py and layer_single_base_pruner.py hook cleanup",
    )
    add_check(
        rows,
        "Text-only calibration exposes attention masks for padding-aware DAS/AMIA",
        unimodal.count("self.temp_encoder_atts = input_tokens.attention_mask.detach()") >= 2,
        "unimodal_prune.py text-only views",
    )
    add_check(
        rows,
        "Runtime smoke checks real-model masks, AMIA scores, scaler, and optional DAS",
        "smoke_tamp_migration_runtime.py" in str(smoke_path)
        and "prepare_calibration_input_encoder" in smoke
        and "check_mask_layout" in smoke
        and "check_cache_attention_layout" in smoke
        and "_encoder_attention_column_scores" in smoke
        and "AdaptiveMultimodalInputActivation" in smoke
        and "args.run_das" in smoke
        and "decoder fallback sparsity is not uniform original sparsity" in smoke,
        "scripts/blip2/smoke_tamp_migration_runtime.py",
    )
    add_check(
        rows,
        "Runtime smoke writes machine-readable evidence with critical fields",
        "parser.add_argument(\"--out_json\"" in smoke
        and "\"mask_summary\": mask_summary" in smoke
        and "\"cache_summary\": cache_summary" in smoke
        and "\"encoder_input_summary\": encoder_input_summary" in smoke
        and "\"amia_score_summary\": amia_score_summary" in smoke
        and "\"amia_selected_rows\": int(wrapped.nsamples)" in smoke
        and "\"das_summary\": das_summary" in smoke
        and "out_path.write_text(json.dumps(summary" in smoke,
        "scripts/blip2/smoke_tamp_migration_runtime.py summary JSON",
    )
    add_check(
        rows,
        "Runtime smoke records encoder input layout and TAMP runtime settings",
        "def summarize_encoder_inputs(" in smoke
        and "\"total_visual_query_tokens\"" in smoke
        and "\"total_valid_text_tokens\"" in smoke
        and "\"total_pad_text_tokens\"" in smoke
        and "\"token_selection\": \"amia\"" in smoke
        and "\"score_method\": \"density_sum\"" in smoke
        and "\"sparsity_ratio_granularity\": \"layer\"" in smoke
        and "\"importance_scope\": \"llm_only\"" in smoke
        and "\"prune_t5\": True" in smoke
        and "\"prune_vit\": False" in smoke
        and "\"max_sparsity_per_layer\": min(1.0, args.sparsity + 0.1)" in smoke,
        "scripts/blip2/smoke_tamp_migration_runtime.py evidence details",
    )
    add_check(
        rows,
        "Runtime smoke fails early with clear missing-path and dependency messages",
        "Missing runtime smoke input path(s)" in smoke
        and "if not os.path.isfile(args.calib_json)" in smoke
        and "if not os.path.isdir(args.images_dir)" in smoke
        and "if not os.path.isfile(args.ckpt)" in smoke
        and "except ModuleNotFoundError as exc" in smoke
        and "requires the full LAVIS runtime" in smoke
        and "environment with PyTorch, Pillow" in smoke
        and "PYTHON_BIN when calling run_tamp_migration_validation.sh" in smoke,
        "scripts/blip2/smoke_tamp_migration_runtime.py early failures",
    )
    add_check(
        rows,
        "Runtime smoke distinguishes raw encoder masks from T5 extended cache masks",
        "cache_mask.dim() >= 3" in smoke
        and "raw_attn.dim() == 2" in smoke
        and "cache_mask.shape[-1] == S" in smoke
        and "extended attention mask does not suppress raw PAD columns" in smoke,
        "scripts/blip2/smoke_tamp_migration_runtime.py cache mask checks",
    )
    add_check(
        rows,
        "Torch-only core smoke checks T5 replay, AMIA, and DAS without importing LAVIS",
        "load_core_namespace" in core_smoke
        and "_normal_t5_block_forward" in core_smoke
        and "das_normal_t5_forward" in core_smoke
        and "decoder cross position bias not propagated" in core_smoke
        and "DAS decoder cross position bias not propagated" in core_smoke
        and "PAD score positions are nonzero" in core_smoke
        and "AMIA scaler_row has non-finite values, likely from padded rows" in core_smoke
        and "AMIA accepted missing attention contribution scores" in core_smoke
        and "DAS language density included padded token" in core_smoke,
        "scripts/blip2/smoke_tamp_core_ops.py",
    )

    add_check(
        rows,
        "Validation runner chains static, core, and optional runtime smoke checks",
        "validate_tamp_migration.py --lavis_root . --out_json" in validation_runner
        and "static_validation.json" in validation_runner
        and "core_smoke.json" in validation_runner
        and "read -r -a PYTHON_CMD" in validation_runner
        and "\"${PYTHON_CMD[@]}\"" in validation_runner
        and "smoke_tamp_core_ops.py" in validation_runner
        and "--out_json \"$CORE_JSON\"" in validation_runner
        and "smoke_tamp_migration_runtime.py" in validation_runner
        and "runtime_smoke.json" in validation_runner
        and "--out_json \"$RUNTIME_JSON\"" in validation_runner
        and "runtime smoke needs all of --calib_json" in validation_runner,
        "scripts/blip2/run_tamp_migration_validation.sh",
    )
    add_check(
        rows,
        "Validation runner checks JSON evidence after static/core and runtime smoke",
        "check_tamp_validation_outputs.py --out_dir \"$OUT_DIR\"" in validation_runner
        and "check_tamp_validation_outputs.py --out_dir \"$OUT_DIR\" --require_runtime" in validation_runner,
        "scripts/blip2/run_tamp_migration_validation.sh JSON evidence checker",
    )
    add_check(
        rows,
        "Validation output checker enforces runtime AMIA/DAS evidence fields",
        "def check_runtime(path: Path)" in validation_checker
        and "\"encoder_input_summary\"" in validation_checker
        and "\"amia_score_summary\"" in validation_checker
        and "\"total_visual_query_tokens\"" in validation_checker
        and "\"invalid_abs_max\"" in validation_checker
        and "\"importance_scope\"" in validation_checker
        and "expected_max = min(1.0, float(data[\"sparsity\"]) + 0.1)" in validation_checker
        and "\"amia_valid_rows_first_batch\"" in validation_checker
        and "\"amia_selected_fraction_first_batch\"" in validation_checker
        and "\"encoder_keys\"" in validation_checker
        and "\"decoder_fallback_keys\"" in validation_checker
        and "args.require_runtime" in validation_checker,
        "scripts/blip2/check_tamp_validation_outputs.py",
    )
    add_check(
        rows,
        "Validation output checker labels static/core/runtime failures",
        "[ERROR][static]" in validation_checker
        and "[ERROR][core]" in validation_checker
        and "[ERROR][runtime]" in validation_checker
        and "\"runtime_required\": bool(args.require_runtime)" in validation_checker,
        "scripts/blip2/check_tamp_validation_outputs.py failure categories",
    )
    add_check(
        rows,
        "Validation runbook documents the real BLIP2-T5 runtime gate",
        "BLIP2-T5 TAMP Migration Validation" in validation_doc
        and "AMIA must use attention contribution scores" in validation_doc
        and "column-wise encoder attention" in validation_doc
        and "max_sparsity_per_layer" in validation_doc
        and "Decoder and non-encoder tensors must fall back to uniform sparsity" in validation_doc
        and "DAS must be computed per T5 encoder `Linear`" in validation_doc
        and "PAD tokens must be excluded" in validation_doc
        and "position_bias" in validation_doc
        and "check_tamp_validation_outputs.py --require_runtime" in validation_doc,
        "scripts/blip2/TAMP_MIGRATION_VALIDATION.md",
    )

    return all(bool(row["passed"]) for row in rows), rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static source checks for the BLIP2-T5 TAMP migration.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lavis_root",
        default=str(Path(__file__).resolve().parents[2]),
        help="Path to the LAVIS root containing evaluate_blip.py.",
    )
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lavis_root = Path(args.lavis_root).resolve()
    ok, rows = validate(lavis_root)

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps({"ok": ok, "checks": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    for row in rows:
        status = "OK" if row["passed"] else "FAIL"
        print(f"[{status}] {row['check']} -- {row['evidence']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
