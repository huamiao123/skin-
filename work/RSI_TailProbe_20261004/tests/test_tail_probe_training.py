"""CPU control-flow acceptance of isolated TailProbe training.

These synthetic fixtures test plumbing, budgets and rejection paths only.
They are never used as method results or as substitutes for real preflight.
"""
import copy
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

from rsi.datasets import collate_multi_reference
from rsi.objectives import per_reference_loss, image_reference_mean
from rsi.runtime import save_checkpoint
from tools import tail_probe_training as training
from tools.tail_probe_preflight import (accumulation_weights, accumulation_control,
                                      _padding_objective_check, _gradient_audit,
                                      _cpu_evidence, _finish_f_reuse, _gradient_comparison)


class ControlDataset(Dataset):
    def __init__(self, count=1471, split="train"):
        self.count,self.split,self.epoch=count,split,0
    def __len__(self): return self.count
    def set_epoch(self, epoch): self.epoch=epoch
    def __getitem__(self,index):
        count=2+index%2
        return dict(image=torch.full((3,1,1),(index+1)/1471),
            masks=torch.tensor([(index+r)%2 for r in range(count)],dtype=torch.float32).reshape(count,1,1,1),
            rater_present=torch.ones(count,dtype=torch.bool),pixel_valid=torch.ones(1,1,1),
            image_id=str(index),group_id=f"fixture-{index}",references=[dict(reference_id=f"r{index}-{r}") for r in range(count)],
            geometry=dict(original_h=1,original_w=1,resized_h=1,resized_w=1,top=0,left=0,size=1),subset="M")


class ControlModel(nn.Module):
    def __init__(self, group="U-NoMessage"):
        super().__init__();self.group=group
        self.register_buffer("fixed_teacher",torch.tensor(0.))
        self.student_tail=nn.Module()
        self.student_tail.register_parameter("weight",nn.Parameter(torch.full((179745,),0.03)))
    def trainable_parameters(self): return (p for p in self.parameters() if p.requires_grad)
    def trainable_parameter_names(self): return [n for n,p in self.named_parameters() if p.requires_grad]
    def frozen_state_hash(self): return hashlib.sha256(self.fixed_teacher.numpy().tobytes()).hexdigest()
    def teacher_state_hash(self): return self.frozen_state_hash()
    def forward_reference(self,rgb): return rgb[:,:1]*0+self.fixed_teacher
    def forward_details(self,rgb,pixel_valid=None):
        ref=self.forward_reference(rgb).detach()
        return dict(z0=ref,z1=ref+rgb[:,:1]*self.student_tail.weight.mean(),message=None)
    def architecture_audit(self):
        return dict(group=self.group,trainable_parameters=179745,
                    trainable_parameter_names=self.trainable_parameter_names())


def cpu_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)


def cpu_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=None)


@pytest.fixture
def installed(monkeypatch,tmp_path):
    torch.set_num_threads(2)
    root=tmp_path/"engineering-fixture";root.mkdir()
    task=root/"task.txt";task.write_text("synthetic fixture; not actual experiment assets")
    teacher=root/"B.pth";teacher.write_text("fixture initializer identity")
    manifest=root/"manifest.csv";manifest.write_text("control fixture only")
    cfg=dict(output_root=str(root),run_root=str(root/"runs"),taskbook=str(task),
        taskbook_sha256=training.sha256_file(task),teacher_B=str(teacher),manifest=str(manifest),
        seed=17,epochs={"D":40},effective_batch=8,microbatch_start=4,amp=False,
        learning_rate=1e-4,weight_decay=1e-4,gradient_clip=1.,gradient_diagnostic_every_epochs=5,
        test_scoring_locked=True,strict_test_scoring_lock=True,eval_batch=4,workers=0,
        input_size=[1,1],cache_dir=str(root/"cache"))
    (root/"preregistration.json").write_text(json.dumps(dict(taskbook_sha256=cfg["taskbook_sha256"],fixture=True)))
    code=dict(source_bundle_hash="fixture-code-hash")
    (root/"code_provenance.json").write_text(json.dumps(code))
    for name in ("asset_audit.json","data_audit.json","environment.json","resolved_paths.json"):
        (root/name).write_text(json.dumps(dict(synthetic_control_flow_fixture=True,file=name)))
    (root/"engineering_cpu_tests.json").write_text(json.dumps(dict(status="PASS",passed_tests=47,synthetic_fixture=True)))
    acceptance=dict(status="PASS",synthetic_control_flow_only=True,
        configuration_sha256=hashlib.sha256(json.dumps(cfg,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest(),
        manifest_sha256=training.sha256_file(manifest),teacher_B_sha256=training.sha256_file(teacher),
        taskbook_sha256=cfg["taskbook_sha256"],code_provenance=code,
        cpu_tests_sha256=training.sha256_file(root/"engineering_cpu_tests.json"))
    (root/"engineering_acceptance.json").write_text(json.dumps(acceptance))
    monkeypatch.setattr(training,"check_inputs",lambda c:(root,training.sha256_file(manifest),training.sha256_file(teacher)))
    monkeypatch.setattr(training,"provenance",lambda c:code)
    monkeypatch.setattr(training,"build_model",lambda c,g:ControlModel(g))
    monkeypatch.setattr(training,"datasets",lambda c:(ControlDataset(),ControlDataset(),ControlDataset(223,"val")))
    original_device_batch=training.device_batch
    monkeypatch.setattr(training,"device_batch",lambda b,d:original_device_batch(b,torch.device("cpu")))
    monkeypatch.setattr(training,"set_seed",cpu_seed)
    monkeypatch.setattr(training,"rng_state",cpu_rng)
    monkeypatch.setattr(training,"environment",lambda:dict(device="CPU synthetic control-flow fixture"))
    monkeypatch.setattr(training,"evaluate",lambda *a,**kw:{**{k:0. for k in training.FIELDS},"dice":0.5,"images":223})
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda:None)
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda:0)
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda:None)
    return cfg,root,code


def signature(cfg,root,code):
    return dict(config=cfg,group="U-NoMessage",seed=17,epochs=40,
        init_checkpoint_hash=training.sha256_file(cfg["teacher_B"]),
        manifest_hash=training.sha256_file(cfg["manifest"]),source_bundle_hash=code["source_bundle_hash"],
        preregistration_sha256=training.sha256_file(root/"preregistration.json"),
        p0_audit_sha256={name:training.sha256_file(root/name) for name in
                         ("asset_audit.json","data_audit.json","environment.json","resolved_paths.json")})


def test_final_seven_images_weights_match_full_image_equal_gradient():
    assert accumulation_weights([4,3]) == [4/7,3/7]
    result=accumulation_control()
    assert result["status"]=="PASS"
    assert result["samples"]==7
    assert accumulation_weights([4,4]) == [0.5,0.5]
    for counts in ([],[0],[4,5],[-1,3]):
        with pytest.raises(ValueError): accumulation_weights(counts)


def test_real_trainer_first_epoch_keeps_1471_images_184_steps_and_normalizes_last_seven(installed,monkeypatch):
    cfg,root,_=installed
    captured=[]
    original_step=training.finish_optimizer_step
    def capture_step(optimizer,scaler,parameters,max_norm):
        captured.append(dict(value=float(parameters[0].detach().mean()),
                             scalar_gradient=float(parameters[0].grad.mean()*parameters[0].numel())))
        return original_step(optimizer,scaler,parameters,max_norm)
    monkeypatch.setattr(training,"finish_optimizer_step",capture_step)
    original_save=training.save_checkpoint
    def stop_at_durable_first_epoch(path,payload):
        original_save(path,payload)
        if Path(path).name=="latest.pth": raise RuntimeError("intentional CPU fixture stop after committed epoch")
    monkeypatch.setattr(training,"save_checkpoint",stop_at_durable_first_epoch)
    with pytest.raises(RuntimeError,match="intentional CPU fixture"):
        training.train_group(cfg,"U-NoMessage")
    assert len(captured)==184
    run=root/"runs/seed17/U-NoMessage"
    latest=torch.load(run/"latest.pth",map_location="cpu")
    assert latest["epoch"]==1
    assert latest["optimizer_step_attempts"]==latest["optimizer_steps"]==184
    assert latest["skipped_optimizer_steps"]==0
    # Last seven examples are shuffled deterministically, rather than taking
    # the final seven manifest indices. Reconstruct the actual loader order.
    dataset=ControlDataset()
    batches=list(training.loader(dataset,4,0,seed=17,epoch=1,shuffle=True))
    samples=[dataset[int(image_id)] for batch in batches[-2:] for image_id in batch["image_id"]]
    assert [len(b["image_id"]) for b in batches[-2:]]==[4,3]
    final=collate_multi_reference(samples)
    value=torch.tensor(captured[-1]["value"],requires_grad=True)
    logits=final["image"][:,:1]*value
    ell=per_reference_loss(logits,final["masks"],final["rater_present"],final["pixel_valid"])
    expected=torch.autograd.grad(image_reference_mean(ell,final["rater_present"]),value)[0]
    assert captured[-1]["scalar_gradient"]==pytest.approx(float(expected),abs=2e-7,rel=2e-5)
    rows=(run/"epoch_diagnostics.csv").read_text().splitlines()
    assert len(rows)==2
    assert latest["selection_rule"].endswith("earlier_tie")


@pytest.mark.parametrize("field",("config","init_checkpoint_hash","manifest_hash","source_bundle_hash","preregistration_sha256","p0_audit_sha256"))
def test_resume_rejects_changed_config_initialization_data_code_and_preregistration(installed,field):
    cfg,root,code=installed
    original=copy.deepcopy(signature(cfg,root,code))
    if field=="config": original[field]["learning_rate"]=0.9
    else: original[field]="changed-identity"
    run=root/"runs/seed17/U-NoMessage";run.mkdir(parents=True)
    save_checkpoint(run/"latest.pth",dict(signature=original))
    with pytest.raises(ValueError,match="Resume config/code/data/init/preregistration mismatch"):
        training.train_group(cfg,"U-NoMessage")


def test_training_refuses_changed_frozen_source_before_any_model_build(installed,monkeypatch):
    cfg,root,_=installed
    (root/"code_provenance.json").write_text(json.dumps(dict(source_bundle_hash="different-code")))
    monkeypatch.setattr(training,"build_model",lambda *a:pytest.fail("model constructed before identity rejection"))
    with pytest.raises(ValueError,match="Code changed"):
        training.train_group(cfg,"U-NoMessage")


def test_training_refuses_missing_engineering_acceptance_or_reused_group(installed):
    cfg,root,code=installed
    acceptance=json.loads((root/"engineering_acceptance.json").read_text())
    acceptance["status"]="CONDITIONAL_FULL_F_REUSE_READOUT_PENDING"
    (root/"engineering_acceptance.json").write_text(json.dumps(acceptance))
    with pytest.raises(ValueError,match="acceptance"):
        training.train_group(cfg,"U-NoMessage")
    acceptance["status"]="PASS"
    (root/"engineering_acceptance.json").write_text(json.dumps(acceptance))
    run=root/"runs/seed17/U-NoMessage";run.mkdir(parents=True)
    (run/"DONE.json").write_text(json.dumps(dict(reused=True)))
    with pytest.raises(ValueError,match="Reused legacy"):
        training.train_group(cfg,"U-NoMessage")


def test_scoring_split_test_fails_before_loading_or_cuda():
    class LockedDataset: split="test"
    with pytest.raises(PermissionError,match="test scoring is locked"):
        training.evaluate(None,LockedDataset(),{})


def test_padding_absent_references_and_fp32_keep_attached_current_gradient():
    z=torch.tensor([[[[0.1,0.2,-0.2,2.]]]],requires_grad=True)
    teacher=torch.tensor([[[[0.,0.1,-0.1,4.]]]])
    batch=dict(masks=torch.tensor([[[[[1.,1.,0.,float("nan")]]],
                                   [[[0.,1.,0.,float("nan")]]]]]),
               rater_present=torch.tensor([[1,1]],dtype=torch.bool),
               pixel_valid=torch.tensor([[[[1.,1.,1.,0.]]]]))
    result=_padding_objective_check(dict(z0=teacher,z1=z),batch)
    assert result["status"]=="PASS"
    assert result["invalid_padding_zero_gradient"]


def test_gradient_audit_rejects_frozen_gradient_and_missing_student_gradient():
    model=ControlModel()
    with pytest.raises(AssertionError,match="missing gradient"):
        _gradient_audit(model)
    batch=collate_multi_reference([ControlDataset()[0],ControlDataset()[1]])
    _,obj,ell,ellref=training.calculate(model,batch)
    assert not ellref.requires_grad
    assert ell.requires_grad
    obj["loss"].backward()
    result=_gradient_audit(model)
    assert result["student_tail"]>0
    model.register_parameter("frozen",nn.Parameter(torch.ones(1),requires_grad=False))
    model.frozen.grad=torch.ones_like(model.frozen)
    with pytest.raises(AssertionError,match="frozen component"):
        _gradient_audit(model)


@pytest.mark.parametrize("field",("configuration_sha256","manifest_sha256","teacher_B_sha256",
                                 "taskbook_sha256","code_provenance","cpu_tests_sha256"))
def test_engineering_acceptance_rejects_unbound_current_identities(installed,field):
    cfg,root,_=installed
    record=json.loads((root/"engineering_acceptance.json").read_text())
    record[field]={"source_bundle_hash":"changed"} if field=="code_provenance" else "changed"
    (root/"engineering_acceptance.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="Engineering acceptance|CPU control-flow acceptance"):
        training.train_group(cfg,"U-NoMessage")


def test_preflight_cpu_evidence_requires_47_passing_tests_and_binds_bytes(installed):
    cfg,root,_=installed
    result=dict(checks={})
    _cpu_evidence(cfg,result)
    assert result["cpu_tests_sha256"]==training.sha256_file(root/"engineering_cpu_tests.json")
    assert result["cpu_control_flow_tests_passed"] is True
    assert result["cpu_control_flow_tests_required"] is False
    (root/"engineering_cpu_tests.json").write_text(json.dumps(dict(status="PASS",passed_tests=46)))
    with pytest.raises(AssertionError,match="tests have not passed"):
        _cpu_evidence(cfg,dict(checks={}))


def mechanical_pass_result(installed):
    cfg,root,_=installed
    result=json.loads((root/"engineering_acceptance.json").read_text())
    result["groups"]=[dict(status="PASS") for _ in range(5)]
    result["legacy_comparisons"]=[dict(status="PASS") for _ in range(8)]
    return result


def test_only_full_historical_numeric_mismatch_can_require_f_rerun(installed):
    cfg,root,_=installed
    def numerical(_): raise ValueError("Historical hard F evaluation differs: F-Mean: fixture mismatch")
    result=_finish_f_reuse(cfg,mechanical_pass_result(installed),numerical)
    assert result["status"]=="REQUIRES_F_RERUN"
    decision=json.loads((root/"reuse_decision.json").read_text())
    assert decision["reused"] is False
    assert decision["same_condition_factors"] is False
    assert decision["core_mechanical_checks"]=="PASS"
    assert (root/"reuse_validation/numerical_mismatch.json").is_file()
    def changed_asset(_): raise ValueError("F reuse identities changed")
    with pytest.raises(ValueError,match="identities changed"):
        _finish_f_reuse(cfg,mechanical_pass_result(installed),changed_asset)
    incomplete=mechanical_pass_result(installed)
    incomplete["legacy_comparisons"][0]["status"]="FAIL"
    with pytest.raises(AssertionError,match="mechanics must pass"):
        _finish_f_reuse(cfg,incomplete,numerical)


def test_full_f_reuse_requires_two_passing_group_readouts(installed):
    cfg,_,_=installed
    passed=_finish_f_reuse(cfg,mechanical_pass_result(installed),
        lambda _:dict(status="PASS",groups=[dict(group=group,references=624,hard_metrics_exact=True)
                                            for group in ("F-Mean","F-RSI")]))
    assert passed["status"]=="PASS"
    with pytest.raises(AssertionError,match="readout not passed"):
        _finish_f_reuse(cfg,mechanical_pass_result(installed),lambda _:dict(status="PASS",groups=[]))


def test_gradient_numerical_comparison_retains_exact_cpu_identity():
    reference={"q_proj.weight":torch.tensor([0.1,-0.2,0.3]),"k_proj.bias":torch.zeros(2)}
    result=_gradient_comparison(reference,{name:value.clone() for name,value in reference.items()})
    assert result["status"]=="PASS"
    assert max(result["maximum_absolute_errors"].values())==0
    assert result["global_vector"]["relative_l2_error"]==0
    assert result["global_vector"]["absolute_norm_difference"]==0
    assert result["global_vector"]["cosine"]==pytest.approx(1.,abs=1e-15)


def test_gradient_numerical_comparison_allows_cuda_roundoff_and_tiny_zero_bias():
    reference={"q_proj.weight":torch.tensor([0.01,-0.02,0.03]),"k_proj.bias":torch.zeros(2)}
    current={name:value.clone() for name,value in reference.items()}
    current["q_proj.weight"][0]+=4.6566e-10
    current["k_proj.bias"][0]=5e-10
    result=_gradient_comparison(reference,current)
    assert result["status"]=="PASS"
    assert result["maximum_absolute_errors"]["k_proj.bias"]>0
    assert result["global_vector"]["relative_l2_error"]<=1e-6
    assert result["tolerance"]==dict(per_tensor_atol=1e-8,per_tensor_rtol=1e-6,
                                      global_vector_relative_l2_max=1e-6)


def test_gradient_numerical_comparison_rejects_material_and_global_vector_errors():
    original={"weight":torch.ones(4)}
    result=_gradient_comparison(original,{"weight":torch.ones(4)+2e-5})
    assert result["status"]=="FAIL"
    # Per-element absolute tolerance alone would permit this zero-ish vector,
    # but its aggregate relative discrepancy violates the second criterion.
    tiny={"weight":torch.full((4,),1e-4)}
    changed={"weight":torch.full((4,),1e-4+5e-9)}
    result=_gradient_comparison(tiny,changed)
    assert result["per_tensor"]["weight"]["allclose"] is True
    assert result["global_vector"]["relative_l2_error"]>1e-6
    assert result["status"]=="FAIL"
    with pytest.raises(AssertionError,match="keys"):
        _gradient_comparison(original,{"different":torch.ones(4)})
    with pytest.raises(AssertionError,match="nonfinite"):
        _gradient_comparison(original,{"weight":torch.full((4,),float("nan"))})
