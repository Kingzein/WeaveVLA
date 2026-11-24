"""Train StarVLA from the base VLM and freshly initialized heads."""
import argparse
import datetime
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from starVLA.training.portable_config import save_portable_config
from .train_samples import W,read_frame,graph_recipe,recipe_digest
from starVLA.model.modules.weavevla.geometry.training_runtime import seed_all,tree_map
from starVLA.model.modules.weavevla.policy.runtime import load_new_model
from .streaming_pool import StreamingPool
DATA_W=W
from starVLA.model.modules.weavevla.geometry.gradient_sync import atomic


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--provisional-legacy-masks',action='store_true')
    ap.add_argument('--preflight-report')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data-root',help='Must equal manifest parent, the root audited by preflight')
    ap.add_argument('--base-config', required=True)
    ap.add_argument('--base-vlm', type=Path, required=True)
    ap.add_argument('--prompt-root', type=Path, required=True)
    ap.add_argument('--action-stats', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=30000)
    ap.add_argument('--slots',type=int,help=argparse.SUPPRESS)
    ap.add_argument('--microbatch',type=int,required=True);ap.add_argument('--output', required=True)
    ap.add_argument('--save-interval',type=int,default=5000)
    ap.add_argument('--zero-config', required=True)
    ap.add_argument('--ddp',action='store_true',help='Not used by ZeRO engine')
    ap.add_argument('--no-grad-checkpoint',action='store_true')
    ap.add_argument('--fast-graph',action='store_true')
    ap.add_argument('--random-offset',action='store_true')
    args=ap.parse_args()
    if args.microbatch < 1 or args.slots not in (None, args.microbatch):
        ap.error('Set a positive per-device batch. Each update consumes one batch per process.')
    args.slots=args.microbatch
    assert args.save_interval > 0 and args.steps > 0
    if args.provisional_legacy_masks: raise ValueError('Repaired pipeline cannot train on legacy masks')
    os.environ['WEAVE_PROMPT_ROOT'] = str(args.prompt_root.resolve())
    os.environ['WEAVE_CALVIN_ACTION_STATS'] = str(args.action_stats.resolve())
    out=Path(args.output).expanduser().resolve()
    rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    if not dist.is_initialized():
        dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20))
    assert dist.get_world_size()==world and dist.get_rank()==rank and dist.get_backend()=='nccl'
    # The optimizer helper uses Accelerate's logger even with a native torchrun launcher.
    from accelerate import PartialState
    PartialState()
    torch.cuda.set_per_process_memory_fraction(.88)
    if rank==0:
        out.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    base_cfg = OmegaConf.load(args.base_config)
    base_cfg.framework.qwenvl.base_vlm = str(args.base_vlm.resolve())
    base_cfg.trainer.pretrained_checkpoint = None
    base_cfg.trainer.is_resume = False
    if rank == 0:
        save_portable_config(base_cfg, out / 'train_config.yaml')

    assert args.random_offset, 'New cache requires phase-aware random offset sampling'
    manifest_path=Path(args.manifest).resolve()
    data_root=Path(args.data_root).resolve() if args.data_root else manifest_path.parent
    if data_root!=manifest_path.parent:
        raise ValueError('Data root differs from manifest root checked by preflight')
    manifest=json.loads(manifest_path.read_text())
    recipe=graph_recipe(manifest)
    audit=data_root/manifest['audit']
    assert audit.parent.name==manifest['geometry_recipe']
    count=manifest['phase0_samples']
    def pool_for_rank(state=None):
        return StreamingPool(audit,rank,world,args.slots,manifest['audit_version'],state)
    streams=pool_for_rank()
    bindings=recipe_digest(recipe)
    seed_all(42)
    model,cfg=load_new_model(base_cfg,recipe=recipe)
    cfg.run_id='weavevla_calvin_d'
    cfg.run_root_dir=str(out.parent)
    cfg.output_dir=str(out)
    model.train()
    model.graph_model.record_trace=not args.fast_graph
    model.qwen_vl_interface.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    if args.no_grad_checkpoint:
        model.qwen_vl_interface.model.gradient_checkpointing_disable()
    cfg.trainer.gradient_checkpointing=not args.no_grad_checkpoint
    cfg.trainer.gradient_sync='deepspeed_zero2_fp32_master'
    cfg.memory_training.optimizer_precision='fp32_master_and_moments'
    cfg.trainer.training_graph_trace=not args.fast_graph
    cfg.trainer.max_train_steps=args.steps
    cfg.trainer.save_interval=args.save_interval
    cfg.trainer.retain_all_model_checkpoints=True
    cfg.datasets.vla_data.per_device_batch_size=args.microbatch
    cfg.trainer.gradient_accumulation_steps=args.slots//args.microbatch
    cfg.memory_training.sequence_microbatch=args.microbatch
    cfg.memory_training.frames_per_gpu_update=args.slots
    cfg.memory_training.sequence_slots=args.slots
    cfg.memory_training.world_size=world
    cfg.memory_training.global_frames_per_update=args.slots*world
    cfg.memory_training.bptt=1
    cfg.memory_training.training_quality='provisional_legacy_masks' if args.provisional_legacy_masks else 'verified_visible_scene'
    cfg.memory_training.mask_repair_pending=args.provisional_legacy_masks
    cfg.memory_training.random_offset=args.random_offset
    cfg.memory_training.offset_range=[0,7] if args.random_offset else [0,0]
    cfg.memory_training.offset_refresh='episode visit boundary'
    cfg.memory_training.offset_seed=42
    cfg.memory_training.observation_manifest=str(manifest_path)
    cfg.memory_training.observation_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    cfg.memory_training.initialization='raw Qwen3-VL-4B; new graph/action heads; no old checkpoint'
    for p in model.parameters():dist.broadcast(p.data,src=0)
    for b in model.buffers():dist.broadcast(b.data,src=0)
    from starVLA.training.trainer_utils.trainer_tools import setup_optimizer_and_scheduler
    optimizer,scheduler=setup_optimizer_and_scheduler(model,cfg)
    import deepspeed
    ds_config=json.loads(Path(args.zero_config).read_text())
    assert ds_config['zero_optimization']['stage']==2 and ds_config['bf16']['enabled']
    ds_config.update(train_micro_batch_size_per_gpu=args.microbatch,
                     gradient_accumulation_steps=1, train_batch_size=args.microbatch*world)
    runner,optimizer,_,scheduler=deepspeed.initialize(model=model,optimizer=optimizer,lr_scheduler=scheduler,config=ds_config,dist_init_required=False)
    assert runner.zero_optimization_stage()==2
    masters=runner.optimizer.single_partition_of_fp32_groups
    assert masters and all(p.dtype==torch.float32 for p in masters)
    seed_all(42 + rank)
    stop=[False]
    def stopping(*_):stop[0]=True
    signal.signal(signal.SIGUSR1,stopping)
    def checkpoint(step):
        if rank == 0:
            temporary = out / f'.step_{step:06d}.tmp'
            temporary.mkdir()
            torch.save(model.state_dict(), temporary / 'model.pt')
            save_portable_config(cfg, temporary / 'config.yaml')
            (temporary / 'geometry_recipe.json').write_text(json.dumps(recipe, indent=2))
            temporary.rename(out / f'step_{step:06d}')
        dist.barrier()

    pool=ThreadPoolExecutor(max_workers=4)
    def prefetch():return [pool.submit(read_frame,r,i,bindings,args.provisional_legacy_masks,data_root) for r,i in streams.peek()]
    pending=prefetch();dist.barrier()
    for step in range(args.steps):
        tic=time.monotonic();torch.cuda.reset_peak_memory_stats();runner.zero_grad()
        t=time.monotonic();batch=[f.result() for f in pending];read_s=time.monotonic()-t
        expected=[r['key'] for r,_ in streams.peek()]
        assert len(set(expected))==args.slots and [ctx['episode_key'] for _,ctx in batch]==expected
        # Advance/prefetch data only: graph memory still corresponds to current batch.
        finished=streams.advance();pending=prefetch()
        flow=geo=0.;t=time.monotonic()
        for first in range(0,args.slots,args.microbatch):
            subset=batch[first:first+args.microbatch]
            model.graph_model.contexts=[ctx for _,ctx in subset]
            values=runner([ex for ex,_ in subset]);loss=values['action_loss']*(args.microbatch/args.slots)
            assert torch.isfinite(loss),f'Nonfinite loss rank{rank} step{step}'
            flow+=float(values['flow_loss'].detach())*(args.microbatch/args.slots)
            geo+=float(values['geometry_loss'].detach())*(args.microbatch/args.slots)
            runner.backward(loss);model.graph_model.detach_memory()
            del values,loss,subset
        torch.cuda.synchronize();compute_s=time.monotonic()-t
        for key in finished:model.graph_model.reset(key)
        assert all(k in set(expected) for k in model.graph_model.states)
        assert all(not x['memory'].requires_grad for s in model.graph_model.states.values() for x in s['registry'].values())
        t=time.monotonic()
        runner.step();torch.cuda.synchronize();sync_s=time.monotonic()-t
        norm=runner.get_global_grad_norm()
        assert np.isfinite(float(norm)), 'Nonfinite global gradient norm'
        stats=torch.tensor([flow,geo],dtype=torch.float64,device='cuda')
        dist.all_reduce(stats);stats/=world
        torch.cuda.synchronize();elapsed=time.monotonic()-tic
        row=dict(step=step+1,rank=rank,audited_on_rank=len(streams.records),episode_visits=sum(s['visit'] for s in streams.state),flow_loss=float(stats[0]),geometry_loss=float(stats[1]),
            grad_norm=float(norm),seconds=elapsed,data_seconds=read_s,compute_seconds=compute_s,
            sync_optimizer_seconds=sync_s,peak_gib=torch.cuda.max_memory_allocated()/2**30,
            epoch=(step+1)*args.slots*world/count,bptt=1,sequence_microbatch=args.microbatch,
            ddp_overlap=False,backend='deepspeed_zero2_fp32_master',gradient_checkpointing=not args.no_grad_checkpoint,
            random_offset=args.random_offset,sampling_offsets=[ctx['sampling_offset'] for _,ctx in batch],
            lr={g.get('name',str(i)):g['lr'] for i,g in enumerate(optimizer.param_groups)},updated_at=time.time())
        row.update(global_batch=args.microbatch*world,world_size=world,geo_weight=.1,source_split='D/training')
        # Upstream evaluates normalized actions on the next TRAINING batch every
        # 100 optimizer updates. No D-validation/test oracle is used here.
        if (step+1)%cfg.trainer.eval_interval==0:
            eval_started=time.monotonic()
            evaluation=[f.result() for f in pending]
            expected_eval=[r['key'] for r,_ in streams.peek()]
            assert [ctx['episode_key'] for _,ctx in evaluation]==expected_eval
            model.graph_model.contexts=[ctx for _,ctx in evaluation]
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                # Upstream's shared evaluation passes PI-only DDIM kwargs that
                # QwenGR00T ignores. GR00T uses its configured four flow steps.
                predicted=model.predict_action(examples=[ex for ex,_ in evaluation])
            # A prediction decorator may use inference_mode. Clone outside that
            # context before the next training forward uses persistent GRU state.
            model.graph_model.states=tree_map(model.graph_model.states,lambda x:x.detach().clone())
            predictions=np.asarray(predicted['normalized_actions'])
            targets=np.asarray([ex['action'] for ex,_ in evaluation])
            assert predictions.shape==targets.shape==(args.microbatch,8,7) and np.isfinite(predictions).all()
            row['mse_score']=float(np.linalg.norm(predictions-targets)/np.prod(targets.shape))
            ended=streams.advance()
            for key in ended:model.graph_model.reset(key)
            pending=prefetch()
            row['action_eval_seconds']=time.monotonic()-eval_started
            del evaluation,predicted,predictions,targets
        atomic(out/f'rank_{rank}_status.json',row)
        if rank==0:
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            atomic(out/'status.json',dict(state='training',**row))
            if (step+1)%cfg.trainer.logging_frequency==0:print('TRAIN',json.dumps(row),flush=True)
        flag=torch.tensor([int(stop[0])],device='cuda');dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        should_stop=bool(flag.item())
        if should_stop or step + 1 == args.steps or (step + 1) % args.save_interval == 0:
            checkpoint(step + 1)
        if should_stop:break
    pool.shutdown(wait=True)
    if rank==0:
        final_state=dict(step=step+1,state='paused' if should_stop else 'complete',updated_at=time.time())
        atomic(out/('paused.json' if should_stop else 'complete.json'),final_state)
        atomic(out/'status.json', final_state)
    dist.barrier();dist.destroy_process_group()
if __name__=='__main__':main()
