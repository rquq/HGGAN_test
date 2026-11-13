"""Stage-specific benchmark CLI; prepare/generate never construct OCR models."""
import argparse
import json

from pipeline.config import load_config


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','generate','train','evaluate','report','preflight','run'])
    parser.add_argument('--config',default='configs/benchmark_32.yml')
    parser.add_argument('--model',choices=['spis_gan','fw_gan','higanplus'])
    parser.add_argument('--arm',choices=['real_only','spis_gan','fw_gan','higanplus'])
    parser.add_argument('--seed',type=int,default=123456)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--time-budget-seconds',type=float)
    parser.add_argument('--stage',choices=['generation','ocr','all'],default='generation')
    parser.add_argument('--gpu-slots',default='0,1')
    parser.add_argument('--previous-output-dir')
    parser.add_argument('--receipt',help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    cfg=load_config(args.config)
    if args.previous_output_dir:
        from pipeline.preflight import restore_outputs
        restore_outputs(args.previous_output_dir,cfg['output_dir'])
    if args.command=='prepare':
        from pipeline.preflight import export_prepared
        result=dict(status='complete',prepared_zip=str(export_prepared(cfg)))
    elif args.command=='generate':
        if not args.model:parser.error('generate requires --model')
        from pipeline.generation import generation_worker
        result=generation_worker(cfg,args.model,args.device,args.time_budget_seconds)
    elif args.command in {'train','evaluate'}:
        if not args.arm:parser.error(args.command+' requires --arm')
        from pipeline.ocr import train,evaluate
        if args.command=='train':
            result=train(cfg,args.arm,args.seed,args.device,args.time_budget_seconds)
        else:
            result=evaluate(cfg,args.arm,args.seed,args.device)
    elif args.command=='report':
        from pipeline.report import report
        result=report(cfg)
    elif args.command=='preflight':
        from pipeline.preflight import check
        result=check(cfg,args.stage)
    else:
        from pipeline.runner import run_pipeline
        result=run_pipeline(cfg,args.config,args.stage,args.gpu_slots.split(','),args.time_budget_seconds)
    if args.receipt:
        from pipeline.common import atomic_json
        atomic_json(args.receipt,result)
    print(json.dumps(result,sort_keys=True))
    return 1 if result.get('status')=='failed' else 0


if __name__=='__main__':
    raise SystemExit(main())
