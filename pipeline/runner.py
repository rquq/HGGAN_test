"""Two independent single-visible-GPU subprocesses; cooperative shared deadline."""
import os
from pathlib import Path
import subprocess
import sys
import time

from .common import atomic_json, read_json


def run_jobs(jobs, output_dir, gpu_slots=('0','1'), budget_seconds=None, poll_seconds=.2):
    slots = [str(slot) for slot in gpu_slots]
    if not slots or len(slots) != len(set(slots)) or any(',' in s or not s for s in slots):
        raise ValueError('GPU slots must be unique single-device identifiers')
    if len({j['id'] for j in jobs}) != len(jobs):
        raise ValueError('job IDs must be unique')
    output = Path(output_dir)
    logs = output/'logs'
    logs.mkdir(parents=True,exist_ok=True)
    deadline = None if budget_seconds is None else time.monotonic()+max(0,budget_seconds)
    pending, active, results = list(jobs), {}, {}
    def remaining():
        return None if deadline is None else max(0.,deadline-time.monotonic())
    def publish():
        rows = [results[j['id']] for j in jobs if j['id'] in results]
        state = ('failed' if any(r['status']=='failed' for r in rows) else
                 'paused' if any(r['status']=='paused' for r in rows) else
                 'running' if pending or active else 'complete')
        receipt = dict(status=state,jobs=rows)
        atomic_json(output/'runner_status.json',receipt)
        return receipt
    try:
        while pending or active:
            for slot in slots:
                if slot in active or not pending:
                    continue
                budget = remaining()
                if budget is not None and budget <= 0:
                    break
                job = pending.pop(0)
                # Never accept an old complete receipt after a worker crash.
                status_path = Path(job['status_path'])
                before = status_path.stat().st_mtime_ns if status_path.exists() else None
                log_path = logs/(job['id']+'.log')
                writer = log_path.open('w',encoding='utf-8')
                argv = list(job['argv'])
                if job.get('budget_arg') and budget is not None:
                    argv += ['--time-budget-seconds',str(budget)]
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=slot, OMP_NUM_THREADS='1',
                           MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                           TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
                try:
                    process = subprocess.Popen(argv,stdout=writer,stderr=subprocess.STDOUT,
                                               env=env,cwd=job.get('cwd'))
                except BaseException:
                    writer.close()
                    raise
                active[slot] = dict(job=job,process=process,writer=writer,
                                    reader=log_path.open(encoding='utf-8',errors='replace'),before=before)
            for slot,item in list(active.items()):
                process, job = item['process'],item['job']
                chunk = item['reader'].read()
                if chunk:
                    print(chunk,end='',flush=True)
                code = process.poll()
                if code is None:
                    continue
                item['writer'].close()
                print(item['reader'].read(),end='',flush=True)
                item['reader'].close()
                path = Path(job['status_path'])
                row = dict(id=job['id'],gpu=slot,returncode=code,status='failed')
                try:
                    if item['before'] is not None and path.stat().st_mtime_ns == item['before']:
                        raise ValueError('stale worker receipt')
                    receipt = read_json(path)
                    if receipt.get('status') not in {'complete','paused','failed'}:
                        raise ValueError('invalid worker status')
                    if code == 0:
                        row['status'] = receipt['status']
                    else:
                        row['error'] = 'worker exited nonzero'
                    row['worker_status'] = receipt
                except (OSError,ValueError,TypeError) as exc:
                    row['error'] = 'missing/invalid worker status: '+type(exc).__name__
                results[job['id']] = row
                del active[slot]
            if pending and deadline is not None and remaining() <= 0:
                for job in pending:
                    results[job['id']] = dict(id=job['id'],status='paused',reason='not_started_deadline')
                pending.clear()
            publish()
            if active:
                time.sleep(poll_seconds)
        return publish()
    except BaseException:
        for item in active.values():
            item['process'].terminate()
        for item in active.values():
            try:
                item['process'].wait(timeout=10)
            except subprocess.TimeoutExpired:
                item['process'].kill();item['process'].wait()
            item['writer'].close();item['reader'].close()
        raise


def run_pipeline(cfg, config_path, stage='generation', gpu_slots=('0','1'), budget_seconds=None):
    from .preflight import check, export_prepared
    from .report import report
    started = time.monotonic()
    check(cfg,stage)
    export_prepared(cfg)
    def remaining():
        return None if budget_seconds is None else max(0,budget_seconds-(time.monotonic()-started))
    root = Path(__file__).resolve().parents[1]
    base = [sys.executable,str(root/'benchmark.py')]
    def job(command,arm,seed=None):
        argv = base+[command,'--config',str(Path(config_path).resolve()),'--device','cuda:0']
        if command == 'generate':
            argv += ['--model',arm]
        else:
            argv += ['--arm',arm,'--seed',str(seed)]
        job_id=f'{command}-{arm}'+('' if seed is None else f'-{seed}')
        receipt=Path(cfg['output_dir'])/'logs'/(job_id+'.status.json')
        argv += ['--receipt',str(receipt)]
        return dict(id=job_id,argv=argv,status_path=str(receipt),budget_arg=True,cwd=str(root))
    receipts=[]
    if stage in {'generation','all'}:
        receipt=run_jobs([job('generate',name) for name in cfg['generation']['models']],
                         cfg['output_dir'],gpu_slots,remaining())
        receipts.append(receipt)
        if receipt['status'] != 'complete':
            report(cfg)
            return receipt
    if stage in {'ocr','all'}:
        jobs=[job('train',arm,int(seed)) for seed in cfg['ocr'].get('seeds',[123456])
              for arm in ['real_only',*cfg['generation']['models']]]
        receipts.append(run_jobs(jobs,cfg['output_dir'],gpu_slots,remaining()))
    report(cfg)
    result=dict(status=receipts[-1]['status'] if receipts else 'complete',stage=stage,stages=receipts)
    atomic_json(Path(cfg['output_dir'])/'runner_status.json',result)
    return result
