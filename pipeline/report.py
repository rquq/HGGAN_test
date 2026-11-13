"""Literal completed-test metrics, with visible incomplete comparison rows."""
import csv
import os
from pathlib import Path
import tempfile

from .common import atomic_json, read_json, sha256_file


def report(cfg):
    output=Path(cfg['output_dir']);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for arm in ['real_only',*cfg['generation']['models']]:
        for seed in cfg['ocr'].get('seeds',[123456]):
            directory=output/'ocr'/arm/f'seed_{seed}'
            row=dict(arm=arm,seed=int(seed),status='not_started',global_step=0)
            if (directory/'status.json').exists():
                status=read_json(directory/'status.json')
                row.update(status=status['status'],global_step=status.get('global_step',0))
                if row['status']=='complete':
                    try:
                        metrics=read_json(directory/'test_metrics.json')
                        identity=read_json(directory/'identity.json')['sha256']
                        if (metrics['identity_sha256'] != identity or status['identity_sha256'] != identity
                                or sha256_file(directory/'test_predictions.jsonl') != metrics['predictions_sha256']):
                            raise ValueError('final receipt mismatch')
                        for key in ['cer_percent','ned_percent','wer_percent','word_accuracy_percent','samples']:
                            row[key]=metrics[key]
                    except (OSError,KeyError,ValueError):
                        row['status']='invalid_final_artifacts'
            if arm != 'real_only':
                path=output/'datasets'/arm/'metadata.json'
                if path.exists():
                    meta=read_json(path)
                    row.update(native_height=meta.get('native_height'),export_height=meta.get('export_height'),
                               export_conversion=meta.get('export_conversion'),precision=meta.get('precision'))
            rows.append(row)
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
        state=read_json(directory/'status.json') if (directory/'status.json').exists() else {}
        meta=read_json(directory/'metadata.json') if (directory/'metadata.json').exists() else {}
        print(' | '.join(str(value) for value in [name,state.get('status',meta.get('status','not_started')),
            state.get('count',meta.get('count',0)),
            f'{meta.get("native_height","—")}→{meta.get("export_height","—")}',
            str(directory/'dataset.zip') if (directory/'dataset.zip').is_file() else '—']))
    print('OCR RESULTS: arm | seed | status | step | CER% | NED% | WER% | word accuracy%')
    for row in rows:
        print(' | '.join(str(row.get(k,'—')) for k in ['arm','seed','status','global_step',
            'cer_percent','ned_percent','wer_percent','word_accuracy_percent']))
    return dict(status='complete',rows=rows)
