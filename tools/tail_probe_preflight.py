"""Real B/data engineering acceptance before any TailProbe full-budget run.

Importing this module does not initialize CUDA or run an experiment. The root
runner alone calls run_preflight so GPU tasks can be serialized. Short-run
weights/optimizers are discarded, never saved as formal initialization.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from rsi.datasets import IMAMultiReferenceDataset, collate_multi_reference, restore_logits
from rsi.models import ControlledModel, state_hash
from rsi.objectives import objective, objective_from_losses, per_reference_loss
from rsi.runtime import atomic_json, environment, set_seed, sha256_file
from rsi.tail_probe_model import GROUPS, EXPECTED_TRAINABLE, TailProbeModel, interpolate_update_logits
from rsi.train import calculate as legacy_calculate
from rsi.train import device_batch, finish_optimizer_step
from tools import tail_probe_training as training


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def accumulation_weights(sample_counts, effective_batch=8):
    """Mechanical oracle for a complete window, including the final 4+3."""
    sample_counts = [int(n) for n in sample_counts]
    total = sum(sample_counts)
    if not sample_counts or any(n <= 0 for n in sample_counts) or not 0 < total <= effective_batch:
        raise ValueError("accumulation window has invalid sample counts")
    return [n / total for n in sample_counts]


def accumulation_control(device="cpu"):
    """Differentiable image-equal check; values are an engineering fixture."""
    # Varied reference counts prevent a reference-equal implementation passing.
    parameter = torch.tensor(0.3, device=device, requires_grad=True)
    values = torch.arange(1, 8, device=device, dtype=torch.float32)
    per_image = (parameter * values - values / 9).square()
    expected = torch.autograd.grad(per_image.mean(), parameter, retain_graph=True)[0]
    weights = accumulation_weights([4, 3])
    observed = torch.autograd.grad(per_image[:4].mean()*weights[0] + per_image[4:].mean()*weights[1], parameter)[0]
    error = float((expected-observed).abs())
    require(torch.allclose(expected, observed, atol=2e-7, rtol=2e-7), "final 7-image accumulation changed image weights")
    return dict(status="PASS", kind="differentiable_control_flow_fixture", samples=7,
                microbatch_samples=[4,3], weights=weights, maximum_gradient_error=error,
                expected_gradient=float(expected), observed_gradient=float(observed),
                final_actual_epoch_samples=1471, attempts_per_full_epoch=184)


def _max_error(a, b):
    require(a.shape == b.shape, "tensor shapes differ")
    require(torch.isfinite(a).all() and torch.isfinite(b).all(), "nonfinite comparison tensor")
    return float((a.detach().float()-b.detach().float()).abs().max())


def _require_exact(a, b, label):
    error = _max_error(a, b)
    require(torch.equal(a, b), f"{label} differs; maximum absolute error={error}")
    return error


GRADIENT_TOLERANCE=dict(per_tensor_atol=1e-8,per_tensor_rtol=1e-6,
                        global_vector_relative_l2_max=1e-6)


def _gradient_comparison(reference, current):
    """CUDA backward numerical acceptance without relaxing forward identity.

    Allclose is checked for each tensor. The complete message gradient vector
    additionally has a strict relative L2 bound; relative bounds on individual
    theoretically zero key biases would turn harmless roundoff into failure.
    The diagnostic reduction runs in CPU float64, independently of training.
    """
    require(set(reference)==set(current) and bool(reference),"message gradient keys differ or are empty")
    names=sorted(reference)
    reference_vectors=[];current_vectors=[];per_tensor={}
    for name in names:
        a,b=reference[name],current[name]
        require(a is not None and b is not None,f"missing message gradient: {name}")
        require(a.shape==b.shape,f"message gradient shape differs: {name}")
        require(torch.isfinite(a).all() and torch.isfinite(b).all(),f"nonfinite message gradient: {name}")
        av=a.detach().cpu().double().reshape(-1)
        bv=b.detach().cpu().double().reshape(-1)
        passed=torch.allclose(bv,av,atol=GRADIENT_TOLERANCE["per_tensor_atol"],
                              rtol=GRADIENT_TOLERANCE["per_tensor_rtol"])
        difference=bv-av
        per_tensor[name]=dict(maximum_absolute_error=float(difference.abs().max()),
            difference_l2_norm=float(difference.norm()),reference_l2_norm=float(av.norm()),
            current_l2_norm=float(bv.norm()),allclose=bool(passed))
        reference_vectors.append(av);current_vectors.append(bv)
    av=torch.cat(reference_vectors);bv=torch.cat(current_vectors)
    difference=bv-av
    reference_norm=float(av.norm());current_norm=float(bv.norm())
    difference_norm=float(difference.norm())
    relative=difference_norm/reference_norm if reference_norm else (0. if difference_norm==0 else None)
    cosine=float(torch.dot(av,bv)/(reference_norm*current_norm)) if reference_norm and current_norm else None
    # Dot/norm roundoff can put an identical FP64 vector a few ulps above 1.
    cosine=max(-1.,min(1.,cosine)) if cosine is not None else None
    passed=all(row["allclose"] for row in per_tensor.values()) and relative is not None and relative<=GRADIENT_TOLERANCE["global_vector_relative_l2_max"]
    return dict(status="PASS" if passed else "FAIL",per_tensor=per_tensor,
        maximum_absolute_errors={name:row["maximum_absolute_error"] for name,row in per_tensor.items()},
        global_vector=dict(parameters=av.numel(),reference_l2_norm=reference_norm,
            current_l2_norm=current_norm,difference_l2_norm=difference_norm,
            relative_l2_error=relative,absolute_norm_difference=abs(current_norm-reference_norm),
            relative_norm_difference=abs(current_norm-reference_norm)/reference_norm if reference_norm else None,
            cosine=cosine,zero_reference_vector=reference_norm==0.),
        tolerance=GRADIENT_TOLERANCE.copy(),diagnostic_arithmetic="CPU float64",
        forward_loss_and_threshold_tolerance="EXACT; unchanged",
        individual_tensor_relative_l2_requirement=False)


def _parameters(model):
    names = model.trainable_parameter_names()
    expected = EXPECTED_TRAINABLE[model.group]
    require(sum(p.numel() for p in model.trainable_parameters()) == expected, "trainable count changed")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4, weight_decay=1e-4)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    optimizer_names = [name for name,p in model.named_parameters() if id(p) in ids]
    require(names == optimizer_names, "optimizer and trainable parameter names differ")
    require(model.independent_tail_storage(), "student and teacher share tail storage")
    return optimizer, dict(trainable_count=expected, trainable_parameter_names=names,
                          optimizer_parameter_names=optimizer_names,
                          independent_student_teacher_storage=True)


def _mode_audit(model, names):
    for mode in (False, True, False, True):
        model.train(mode)
        require(model.trainable_parameter_names() == names, "mode switch changed trainable set")
        for name, module in model._components().items():
            wanted = mode and model._component_trainable(name)
            require(all(child.training == wanted for child in module.modules()), f"component mode changed: {name}")
            require(all(p.requires_grad == model._component_trainable(name) for p in module.parameters()),
                    f"component trainability changed: {name}")


def _gradient_audit(model):
    totals = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            require(parameter.grad is not None, f"missing gradient: {name}")
            require(torch.isfinite(parameter.grad).all(), f"nonfinite gradient: {name}")
            component = name.split(".",1)[0]
            totals[component] = totals.get(component,0.) + float(parameter.grad.float().square().sum())
        else:
            require(parameter.grad is None, f"frozen component received gradient: {name}")
    require(all(v > 0 for v in totals.values()), "a trainable component has zero total gradient")
    return {name:math.sqrt(value) for name,value in totals.items()}


def _small_datasets(cfg):
    common = dict(manifest=cfg["manifest"], seed=17, size=cfg["input_size"][0], cache_dir=cfg["cache_dir"])
    all_train = IMAMultiReferenceDataset(split="train",subset="M",train=True,**common)
    all_val = IMAMultiReferenceDataset(split="val",subset="M",train=False,**common)
    require(len(all_train) == 1471 and len(all_val) == 223, "preflight cohort sizes changed")
    train_ids = [refs[0]["image_id"] for refs in all_train.samples[:16]]
    # Keep selection independent of labels and model outcomes; mixed counts
    # provide natural collate padding when available among the first images.
    val_ids = [refs[0]["image_id"] for refs in all_val.samples[:4]]
    train = IMAMultiReferenceDataset(split="train",subset="M",train=True,image_ids=train_ids,**common)
    canonical = IMAMultiReferenceDataset(split="train",subset="M",train=False,image_ids=train_ids,**common)
    val = IMAMultiReferenceDataset(split="val",subset="M",train=False,image_ids=val_ids,**common)
    train.set_epoch(1)
    cpu_train = [train[i] for i in range(len(train))]
    cpu_val = [val[i] for i in range(len(val))]
    for i,sample in enumerate(cpu_train):
        again = train[i]
        for name in ("image","masks","pixel_valid"):
            require(torch.equal(sample[name],again[name]), "deterministic joint augmentation changed")
    for sample in cpu_val:
        geometry = sample["geometry"]
        for ref in sample["references"]:
            with Image.open(ref["mask_path"]) as handle:
                require(handle.size == (geometry["original_w"],geometry["original_h"]), "mask/geometry mismatch")
        valid = sample["pixel_valid"].bool()
        require(valid.any(), "empty real valid area")
        valid_masks=sample["masks"][:,0][:,valid[0]]
        require(((valid_masks==0)|(valid_masks==1)).all(),
                "canonical masks are not binary in valid area")
    return train,canonical,val,cpu_train,cpu_val


def _padding_objective_check(out, batch):
    z = out["z1"].detach().float().clone().requires_grad_(True)
    ref = out["z0"].detach().float()
    masks,present,valid = batch["masks"],batch["rater_present"],batch["pixel_valid"]
    original = objective(z,ref,masks,present,valid,method="rsi",weight=3)
    original_grad = torch.autograd.grad(original["loss"],z)[0]
    modified = z.detach().clone()
    modified[~valid.bool()] = float("nan")
    modified.requires_grad_(True)
    extra_masks = torch.cat([masks,torch.full_like(masks[:,:1],float("nan"))],dim=1)
    extra_present = torch.cat([present,torch.zeros_like(present[:,:1])],dim=1)
    changed = objective(modified,ref,extra_masks,extra_present,valid,method="rsi",weight=3)
    changed_grad = torch.autograd.grad(changed["loss"],modified)[0]
    loss_error = _max_error(original["loss"],changed["loss"])
    grad_error = _max_error(original_grad,changed_grad)
    require(loss_error<=2e-7 and grad_error<=2e-7,"missing/padding handling changed loss or gradient")
    require(torch.count_nonzero(changed_grad[~valid.bool()])==0, "invalid padding received gradient")
    return dict(status="PASS", objective_max_error=loss_error, gradient_max_error=grad_error,
                invalid_padding_zero_gradient=True, appended_absent_reference_ignored=True)


def _legacy_comparison(cfg, base, cpu_train, cpu_val, *, evidence=None):
    evidence = [] if evidence is None else evidence
    for group in ("F-Mean","F-RSI"):
        checkpoint = cfg["selected_checkpoints"][group]
        saved = torch.load(checkpoint,map_location="cpu")
        original = ControlledModel(pretrained=False,input_size=cfg["input_size"][0])
        original.load_state_dict(saved["model"],strict=True)
        original.set_stage("D").cuda()
        model = TailProbeModel(base,group)
        model.load_legacy_state_dict(saved["model"],strict=True)
        model.cuda()
        del saved
        before = model.frozen_state_hash()
        original_before=original.frozen_state_hash()
        original_full_state_before=state_hash(original)
        for label,samples in (("augmented_train",cpu_train[:4]),("canonical_val",cpu_val)):
            batch = device_batch(collate_multi_reference(samples),torch.device("cuda"))
            for amp in (False,True):
                model.train(label=="augmented_train");original.train(label=="augmented_train")
                model.zero_grad(set_to_none=True);original.zero_grad(set_to_none=True)
                with torch.autocast("cuda",enabled=amp):
                    prior,prior_obj,prior_ell,prior_ref = legacy_calculate(original,batch,"D",
                        "rsi" if group.endswith("RSI") else "d0",3. if group.endswith("RSI") else 0.)
                    current,current_obj,current_ell,current_ref = training.calculate(model,batch)
                errors = {name:_require_exact(prior[name],current[name],f"legacy {group}/{label}/amp={amp}/{name}")
                          for name in ("z0","z1","message")}
                for name in ("loss","seg","risk","weighted_risk","rho","J"):
                    errors[name] = _require_exact(prior_obj[name],current_obj[name],f"legacy {name}")
                errors["per_reference_loss"] = _require_exact(prior_ell,current_ell,"legacy per-reference loss")
                errors["teacher_loss"] = _require_exact(prior_ref,current_ref,"legacy teacher loss")
                require(torch.equal(prior["z1"]>=0,current["z1"]>=0),"legacy canonical threshold masks differ")
                for i,geometry in enumerate(batch["geometry"]):
                    a=restore_logits(prior["z1"][i].float(),geometry)
                    b=restore_logits(current["z1"][i].float(),geometry)
                    _require_exact(a,b,"legacy restored logits")
                    require(torch.equal(a>=0,b>=0),"legacy original-coordinate masks differ")
                prior_obj["loss"].backward(retain_graph=True)
                _gradient_audit(original)
                original_grad_first={name:p.grad.detach().clone() for name,p in original.message.named_parameters()}
                original.zero_grad(set_to_none=True)
                # Reuse the identical objective graph without parameter updates
                # to observe backward reduction noise inside the old model.
                prior_obj["loss"].backward()
                original_grad_repeat={name:p.grad.detach().clone() for name,p in original.message.named_parameters()}
                current_obj["loss"].backward()
                current_grad={name:p.grad.detach() for name,p in model.message.named_parameters()}
                gradient_comparison=_gradient_comparison(original_grad_first,current_grad)
                within_implementation_noise=_gradient_comparison(original_grad_first,original_grad_repeat)
                gradient_errors=gradient_comparison["maximum_absolute_errors"]
                _gradient_audit(model)
                require(model.frozen_state_hash()==before,"legacy comparison altered frozen state")
                require(original.frozen_state_hash()==original_before,"repeated legacy backward altered frozen state")
                require(state_hash(original)==original_full_state_before,"repeated legacy backward altered model parameters/buffers")
                evidence.append(dict(group=group,batch=label,images=len(samples),amp=amp,
                    checkpoint_path=str(checkpoint),checkpoint_sha256=sha256_file(checkpoint),
                    maximum_absolute_errors=errors,message_gradient_maximum_absolute_errors=gradient_errors,
                    gradient_comparison=gradient_comparison,
                    within_old_implementation_repeated_backward=within_implementation_noise,
                    repeated_backward_optimizer_steps=0,repeated_backward_model_state_hash=original_full_state_before,
                    repeated_backward_model_parameters_and_buffers_unchanged=True,
                    canonical_and_original_threshold_masks_exact=True,strict_state_load=True,
                    status="PASS" if gradient_comparison["status"]==within_implementation_noise["status"]=="PASS" else "FAIL"))
                require(gradient_comparison["status"]=="PASS",
                        f"legacy gradient numerical acceptance failed: {group}/{label}/amp={amp}: {gradient_comparison}")
                require(within_implementation_noise["status"]=="PASS",
                        f"repeated legacy backward numerical stability failed: {group}/{label}/amp={amp}: {within_implementation_noise}")
                del prior,current,prior_obj,current_obj,prior_ell,current_ell,prior_ref,current_ref
                del original_grad_first,original_grad_repeat,current_grad
        del model,original,batch;gc.collect();torch.cuda.empty_cache()
    return evidence


def _cpu_evidence(cfg, result):
    path=Path(cfg["output_root"])/"engineering_cpu_tests.json"
    evidence=json.loads(path.read_text())
    require(evidence.get("status")=="PASS" and evidence.get("passed_tests",0)>=47,
            "necessary CPU mechanical/control-flow tests have not passed")
    result["cpu_tests_sha256"]=sha256_file(path)
    result["checks"]["cpu_mechanical_and_control_flow_tests"]=dict(
        status="PASS",path=str(path),sha256=result["cpu_tests_sha256"],
        passed_tests=evidence["passed_tests"],evidence=evidence)
    result["cpu_control_flow_tests_required"]=False
    result["cpu_control_flow_tests_passed"]=True


def _finish_f_reuse(cfg, result, prove):
    """Only readout numerical mismatch permits rebuilding the two F groups.

    Missing weights, changed cohorts/identity, cache failures and mechanical
    mismatches still propagate as FAIL. The two explicit historical numeric
    mismatch messages originate in the audited prove_f_reuse implementation.
    """
    require(len(result.get("groups",[]))==5 and
            all(row.get("status")=="PASS" for row in result["groups"]),
            "all group mechanical checks must pass before F reuse")
    require(len(result.get("legacy_comparisons",[]))==8 and
            all(row.get("status")=="PASS" for row in result["legacy_comparisons"]),
            "legacy forward/loss/gradient mechanics must pass before F reuse")
    try:
        reuse=prove(cfg)
    except (ValueError,AssertionError) as error:
        numerical_mismatch=str(error).startswith(("Historical hard F evaluation differs:",
                                                 "Historical canonical F loss differs:"))
        if not numerical_mismatch:
            raise
        decision=dict(status="REQUIRES_F_RERUN",reused=False,same_condition_factors=False,
            reason=str(error),failure_type=type(error).__name__,traceback=traceback.format_exc(),
            core_mechanical_checks="PASS",historical_f_readout="NUMERICAL_MISMATCH",
            required_action="Reinitialize both F groups from common B and train independent full 40 epochs; retain historical differences",
            configuration_sha256=result["configuration_sha256"],
            manifest_sha256=result["manifest_sha256"],teacher_B_sha256=result["teacher_B_sha256"],
            taskbook_sha256=result["taskbook_sha256"],
            source_bundle_hash=result["code_provenance"]["source_bundle_hash"],
            cpu_tests_sha256=result["cpu_tests_sha256"],test_scoring_locked=True)
        atomic_json(Path(cfg["output_root"])/"reuse_decision.json",decision)
        atomic_json(Path(cfg["output_root"])/"reuse_validation"/"numerical_mismatch.json",decision)
        result["full_f_reuse_readout"]=decision
        result["status"]="REQUIRES_F_RERUN"
        return result
    readouts=reuse.get("groups",[])
    require(reuse.get("status")=="PASS" and len(readouts)==2 and
            {row.get("group") for row in readouts}=={"F-Mean","F-RSI"} and
            all(row.get("references")==624 and row.get("hard_metrics_exact") is True for row in readouts),
            "full historical F readout not passed")
    result["full_f_reuse_readout"]=reuse
    result["status"]="PASS"
    return result


def run_preflight(cfg):
    """Run real checkpoint/data GPU acceptance and full legacy-F val reuse.

    Mechanical or identity exceptions preserve checks and a FAIL record, then
    propagate. A historical full-readout numerical mismatch alone returns
    REQUIRES_F_RERUN. PASS requires the full 624-reference readout for both F.
    """
    root=Path(cfg["output_root"])
    target=root/"engineering_acceptance.json"
    started=time.monotonic()
    result=dict(status="RUNNING",started_at=datetime.now(timezone.utc).isoformat(),
                configuration_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
                taskbook_sha256=sha256_file(cfg["taskbook"]),test_scoring_locked=True,
                short_run_weights_discarded=True,formal_initialization="strict selected common B only",
                legacy_gradient_numerical_tolerance=GRADIENT_TOLERANCE.copy(),
                checks={},groups=[],legacy_comparisons=[])
    atomic_json(target,result)
    try:
        _,manifest_hash,init_hash=training.check_inputs(cfg)
        _cpu_evidence(cfg,result)
        require(torch.cuda.is_available(),"CUDA unavailable: real preflight cannot pass")
        for filename in ("asset_audit.json","data_audit.json"):
            audit=json.loads((root/filename).read_text())
            require(audit.get("status")=="PASS",f"required audit has not passed: {filename}")
        result.update(manifest_sha256=manifest_hash,teacher_B_sha256=init_hash,
                      environment=environment(),code_provenance=training.provenance(cfg))
        set_seed(17);torch.set_num_threads(4)
        result["runtime_numerics"]=dict(deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            rule="unchanged original set_seed(17); only gradient diagnostic acceptance has a numerical tolerance")
        train,canonical,val,cpu_train,cpu_val=_small_datasets(cfg)
        result["fixed_batches"]=dict(train_image_ids=[s["image_id"] for s in cpu_train],
            val_image_ids=[s["image_id"] for s in cpu_val],augmentation_epoch=1,
            train_reference_counts=[len(s["references"]) for s in cpu_train],
            val_reference_counts=[len(s["references"]) for s in cpu_val],
            selection="first sorted 16 train and 4 val; no label/performance selection",
            repeated_augmented_inputs_exact=True)
        initial=torch.load(cfg["teacher_B"],map_location="cpu")
        base=ControlledModel(pretrained=False,input_size=cfg["input_size"][0])
        base.load_state_dict(initial["model"],strict=True);base.set_stage("B").eval()
        del initial
        fixed=device_batch(collate_multi_reference(cpu_val),torch.device("cuda"))
        common_on,common_ref=None,None
        for group in GROUPS:
            tick=time.monotonic();torch.cuda.reset_peak_memory_stats()
            model=TailProbeModel(base,group).cuda()
            optimizer,param_audit=_parameters(model)
            names=param_audit["trainable_parameter_names"]
            _mode_audit(model,names)
            frozen_hash=model.frozen_state_hash();teacher_hash=model.teacher_state_hash()
            model.eval()
            with torch.no_grad(),torch.autocast("cuda",enabled=False):
                reference,on=model.forward_pair(fixed["image"],fixed["pixel_valid"])
                off=model.forward_off(fixed["image"],fixed["pixel_valid"])
            off_error=_require_exact(off,reference,"initial current off vs B teacher")
            if common_ref is None:
                common_ref=reference.detach().cpu();common_on=on.detach().cpu()
            reference_error=_require_exact(reference.cpu(),common_ref,"common teacher across groups")
            on_error=_require_exact(on.cpu(),common_on,"common B on across message groups") if group!="U-NoMessage" else None
            if group=="U-NoMessage": _require_exact(on,reference,"NoMessage initial output vs teacher")
            for alpha,expected in ((0.,reference),(1.,on)):
                _require_exact(interpolate_update_logits(reference,on,alpha),expected.float(),f"alpha {alpha} endpoint")
            for i,geometry in enumerate(fixed["geometry"]):
                require(restore_logits(on[i],geometry).shape==(geometry["original_h"],geometry["original_w"]),
                        "coordinate restoration changed mask size")
            result["checks"]["padding_and_absent_references"]=_padding_objective_check({"z0":reference,"z1":on},fixed)
            fixed_teacher=reference.detach().clone()
            del reference,on,off
            hooks=[]
            if group=="U-NoMessage":
                def forbidden(*_): raise AssertionError("U-NoMessage executed Transformer or message")
                hooks=[module.register_forward_pre_hook(forbidden) for module in (model.transformer,model.message)]
            attempted,actual,skipped=0,0,0
            gradient_norms={}
            try:
                scaler=torch.cuda.amp.GradScaler(enabled=cfg["amp"])
                model.train();optimizer.zero_grad(set_to_none=True)
                # F only needs a gradient probe. U groups exercise two complete
                # 8-image optimizer attempts on actual augmented train images.
                count=16 if group.startswith("U-") else 4
                for start in range(0,count,4):
                    batch=device_batch(collate_multi_reference(cpu_train[start:start+4]),torch.device("cuda"))
                    with torch.autocast("cuda",enabled=cfg["amp"]):
                        out,obj,_,_=training.calculate(model,batch)
                    scale=(len(batch["image_id"])/8 if group.startswith("U-") else 1.)
                    scaler.scale(obj["loss"]*scale).backward()
                    gradient_norms=_gradient_audit(model)
                    if group.startswith("U-") and (start//4+1)%2==0:
                        info=finish_optimizer_step(optimizer,scaler,list(model.trainable_parameters()),cfg["gradient_clip"])
                        attempted+=1;actual+=int(not info["skipped"]);skipped+=int(info["skipped"])
                        require(math.isfinite(info["norm"]),"preflight optimizer encountered nonfinite gradient")
                        optimizer.zero_grad(set_to_none=True)
                    del out,obj,batch
                if group.startswith("U-"):
                    require(attempted==2 and actual==2,"short-run attempts or actual updates changed")
                optimizer.zero_grad(set_to_none=True)
                eval_metrics=training.evaluate(model,val,{**cfg,"workers":0})
                diagnostic=training.gradient_diagnostic(model,fixed)
                require(model.trainable_parameter_names()==names,"evaluation/diagnostic changed trainable set")
                _mode_audit(model,names)
                with torch.no_grad(),torch.autocast("cuda",enabled=False):
                    now=model.forward_reference(fixed["image"])
                teacher_error=_require_exact(now,fixed_teacher,"post-step FP32 teacher")
                require(model.teacher_state_hash()==teacher_hash,"teacher parameter/buffer hash drift")
                require(model.frozen_state_hash()==frozen_hash,"frozen encoder/parameter/buffer hash drift")
            finally:
                for hook in hooks: hook.remove()
            evidence=dict(group=group,status="PASS",**param_audit,
                teacher_hash=teacher_hash,frozen_hash=frozen_hash,initial_on_error=on_error,
                initial_reference_error=reference_error,initial_off_teacher_error=off_error,
                final_fp32_teacher_error=teacher_error,all_frozen_hashes_stable=True,
                gradient_component_norms=gradient_norms,
                gradient_component_norms_are_amp_scaled=bool(cfg["amp"]),optimizer_attempts=attempted,
                actual_optimizer_updates=actual,amp_skipped_updates=skipped,
                diagnostic=diagnostic,small_val_evaluation=eval_metrics,
                no_message_transformer_and_message_forbidden=group=="U-NoMessage",
                peak_cuda_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-tick)
            result["groups"].append(evidence);atomic_json(target,result)
            print(f"TailProbe preflight {group}: PASS, attempts={attempted}, updates={actual}, teacher error=0",flush=True)
            del model,optimizer,scaler,fixed_teacher,now;gc.collect();torch.cuda.empty_cache()
        result["legacy_comparisons"]=_legacy_comparison(cfg,base,cpu_train,cpu_val,evidence=result["legacy_comparisons"])
        result["checks"]["last_7_image_accumulation"]=accumulation_control("cuda")
        try:
            IMAMultiReferenceDataset("unread_manifest",split="test")
        except PermissionError: pass
        else: raise AssertionError("dataset test lock failed")
        class LockedDataset: split="test"
        try:
            training.evaluate(None,LockedDataset(),cfg)
        except PermissionError: pass
        else: raise AssertionError("evaluation test lock failed")
        result["checks"]["test_scoring_entrypoints"]=dict(status="PASS",dataset_rejected=True,evaluate_rejected=True)
        result["status"]="CONDITIONAL_FULL_F_REUSE_READOUT_PENDING"
        atomic_json(target,result)
        del base,train,canonical,val,fixed,cpu_train,cpu_val;gc.collect();torch.cuda.empty_cache()
        from tools.tail_probe_analysis import prove_f_reuse
        result=_finish_f_reuse(cfg,result,prove_f_reuse)
        result["finished_at"]=datetime.now(timezone.utc).isoformat()
        result["wall_seconds"]=time.monotonic()-started
        atomic_json(target,result)
        return result
    except Exception as error:
        result.update(status="FAIL",failure_type=type(error).__name__,failure=str(error),
                      traceback=traceback.format_exc(),wall_seconds=time.monotonic()-started,
                      finished_at=datetime.now(timezone.utc).isoformat())
        atomic_json(target,result)
        raise
