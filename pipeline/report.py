"""Literal completed-test metrics, with visible incomplete comparison rows."""
import csv
import math
import os
from pathlib import Path
import tempfile

from .common import atomic_json, digest_json, read_json, sha256_file
from .config import ocr_settings
from .identities import ocr_payload
from .manifest import verify_prepared


def _receipt(path):
    if not path.exists():
        return {}
    try:
        value=read_json(path)
        if not isinstance(value,dict):
            raise ValueError('receipt must be a mapping')
        return value
    except (OSError,ValueError):
        return dict(status='incompatible',reason='invalid saved receipt: '+path.name)


def _completed_metrics(cfg,arm,seed,directory,status):
    """Read-only current-experiment gate; never prepares or binds identities."""
    prepared=verify_prepared(cfg)
    synthetic=None
    if arm!='real_only':
        from .generation import verify_dataset
        synthetic=verify_dataset(cfg,arm,read_only=True)
    expected=ocr_payload(cfg,arm,seed,prepared,synthetic)
    saved=read_json(directory/'identity.json')
    identity=digest_json(expected)
    if saved!=dict(sha256=identity,payload=expected):
        raise ValueError('saved OCR experiment differs from current cohort/test/recipe/arm/seed')
    options=ocr_settings(cfg)
    metrics=read_json(directory/'test_metrics.json')
    if (status.get('identity_sha256')!=identity or status.get('global_step')!=options['max_steps']
            or metrics.get('identity_sha256')!=identity
            or metrics.get('selection')!='validation_cer_wer'
            or sha256_file(directory/'test_predictions.jsonl')!=metrics['predictions_sha256']):
        raise ValueError('incomplete/incompatible final test receipt')
    step=metrics['checkpoint_step']
    if (isinstance(step,bool) or not isinstance(step,int) or not 0<step<=options['max_steps']
            or metrics['checkpoint']!=f'checkpoint-{step}'):
        raise ValueError('invalid selected checkpoint receipt')
    for key in ('cer_percent','ned_percent','wer_percent','word_accuracy_percent'):
        if isinstance(metrics[key],bool) or not isinstance(metrics[key],(int,float)) or not math.isfinite(metrics[key]):
            raise ValueError('invalid final metric')
    if metrics.get('samples')!=read_json(prepared/'metadata.json')['test_count']:
        raise ValueError('final test population mismatch')
    return metrics,synthetic


def _upstream_fields(metadata):
    from generators.external import upstream_provenance
    source=(metadata or {}).get('model_provenance',{})
    upstream=upstream_provenance(source)
    fields={'upstream_'+key:value for key,value in upstream.items()}
    fields['upstream_assurance']=('unknown_exploratory' if 'unknown' in upstream.values()
                                  else 'declared_no_test_training')
    return fields


def report(cfg):
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arm in ['real_only',*cfg['generation']['models']]:
        for seed in cfg['ocr'].get('seeds',[123456]):
            directory=output/'ocr'/arm/f'seed_{seed}'
            row=dict(arm=arm,seed=int(seed),status='not_started',global_step=0,
                     real_count=cfg['dataset']['real_count'],synthetic_count=0 if arm=='real_only' else cfg['generation']['count'],
                     split_seed=cfg['split_seed'],generation_seed=cfg['generation']['seed'],
                     wall_time_seconds='unknown',peak_gpu_memory_mib='unknown')
            metadata=None
            if (directory/'status.json').exists():
                status=_receipt(directory/'status.json')
                row.update(status=status.get('status','incompatible'),global_step=status.get('global_step',0))
                if 'reason' in status:row['reason']=status['reason']
                if row['status']=='complete':
                    try:
                        metrics,metadata=_completed_metrics(cfg,arm,int(seed),directory,status)
                        for key in ['cer_percent','ned_percent','wer_percent','word_accuracy_percent','samples']:
                            row[key]=metrics[key]
                        row.update(selected_checkpoint=metrics['checkpoint'],selected_update=metrics['checkpoint_step'])
                    except (OSError,KeyError,ValueError,TypeError) as exc:
                        row['status']='incompatible'
                        row['reason']=str(exc)
            if arm != 'real_only':
                path=output/'datasets'/arm/'metadata.json'
                if path.exists():
                    meta=metadata or _receipt(path)
                    if meta.get('status')=='incompatible':
                        row.update(status='incompatible',reason=meta.get('reason','invalid generation receipt'))
                    row.update(native_height=meta.get('native_height'),export_height=meta.get('export_height'),
                               export_conversion=meta.get('export_conversion'),precision=meta.get('precision'))
                    try:
                        row.update(_upstream_fields(meta))
                    except (ValueError,TypeError):
                        source=meta.get('model_provenance',{})
                        declared=source.get('upstream_provenance',{}) if isinstance(source,dict) else {}
                        declared=declared if isinstance(declared,dict) else {}
                        row['upstream_assurance']='rejected_or_invalid_declaration'
                        row['upstream_test_data_training']=declared.get('test_data_training','unknown')
                else:
                    row.update(_upstream_fields(None))
            rows.append(row)
    controls={row['seed']:row for row in rows if row['arm']=='real_only' and row['status']=='complete'}
    for row in rows:
        if row['arm']!='real_only' and row['status']=='complete' and row['seed'] in controls:
            control=controls[row['seed']]
            for key in ('cer','ned','wer','word_accuracy'):
                row[key+'_delta_pp_vs_real_only']=row[key+'_percent']-control[key+'_percent']
    atomic_json(output/'results.json',dict(rows=rows))
    keys=list(dict.fromkeys(key for row in rows for key in row))
    fd,temporary=tempfile.mkstemp(prefix='.results-',dir=output)
    try:
        with os.fdopen(fd,'w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(rows)
        os.replace(temporary,output/'results.csv')
    finally:
        if os.path.exists(temporary):os.unlink(temporary)
    print('GENERATION RESULTS: model | status | count | native→export | ZIP')
    for name in cfg['generation']['models']:
        directory=output/'datasets'/name
        state=_receipt(directory/'status.json')
        meta=_receipt(directory/'metadata.json')
        print(' | '.join(str(value) for value in [name,state.get('status',meta.get('status','not_started')),
            state.get('count',meta.get('count',0)),
            f'{meta.get("native_height","—")}→{meta.get("export_height","—")}',
            str(directory/'dataset.zip') if (directory/'dataset.zip').is_file() else '—']))
    print('OCR RESULTS: arm | seed | status | step | CER% | NED% | WER% | word accuracy%')
    for row in rows:
        print(' | '.join(str(row.get(k,'—')) for k in ['arm','seed','status','global_step',
            'cer_percent','ned_percent','wer_percent','word_accuracy_percent']))
    return dict(status='complete',rows=rows)
