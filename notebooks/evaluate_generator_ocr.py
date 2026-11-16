"""Independent, full-split same-transcript OCR for frozen baseline generators."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time


def levenshtein(a, b):
    """Character edit distance without adding a runtime dependency."""
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(temporary, path)


def selected_rows(texts, alphabet, policy, limit):
    if policy not in ('error', 'common_subset'):
        raise ValueError('unsupported_policy must be error or common_subset')
    unsupported = [i for i, text in enumerate(texts) if not text or not set(text) <= set(alphabet)]
    if unsupported and policy == 'error':
        raise ValueError(f'{len(unsupported)} rows contain unsupported/empty targets; examples {unsupported[:10]}. '
                         'Use explicit common_subset policy only for a labeled common-alphabet comparison.')
    blocked = set(unsupported)
    rows = [i for i in range(len(texts)) if i not in blocked]
    if limit is not None:
        if limit < 1:
            raise ValueError('limit must be positive or null')
        rows = rows[:limit]
    if not rows:
        raise ValueError('No evaluable rows')
    return rows, unsupported


def run(cfg, model_name, output):
    import hashlib
    import random
    import numpy as np
    from PIL import Image
    import torch
    import yaml
    torch.set_num_threads(int(cfg.get('cpu_threads', 2)))
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    sys.path.insert(0, cfg['benchmark_root'])
    from generators.external import read_model_config
    from pipeline.data import IAMCorpus
    from pipeline.common import sha256_file
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    profile = read_model_config(Path(cfg['benchmark_root'])/'configs/generators'/f'{model_name}.yml')
    profile.update(cfg['models'][model_name])
    # All arms use the intersection before any model-specific generation.
    alphabets = []
    for name in cfg['models']:
        p = cfg['models'][name]
        alphabet_file = Path(p['source_root'])/'lib/alphabet.py'
        import ast
        tree = ast.parse(alphabet_file.read_text())
        # Parse only the pinned data literal; never execute a model package.
        assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'Alphabets' for t in n.targets))
        alphabets.append(set(ast.literal_eval(assignment.value)['all']))
    common = set.intersection(*alphabets)
    device = cfg['device']
    adapter = importlib.import_module('generators.'+model_name).load_adapter(profile, device)
    processor = TrOCRProcessor.from_pretrained(cfg['ocr_model'], revision=cfg['ocr_revision'],
                                              local_files_only=not cfg['allow_download'])
    reader = VisionEncoderDecoderModel.from_pretrained(cfg['ocr_model'], revision=cfg['ocr_revision'],
                                                       local_files_only=not cfg['allow_download']).float().eval().requires_grad_(False).to(device)
    totals = {name: dict(samples=0, char_edits=0, characters=0, wrong_words=0)
              for name in (['generated', 'real'] if cfg['real_control'] else ['generated'])}

    def recognize(images):
        with torch.inference_mode():
            pixels = processor(images=images, return_tensors='pt').pixel_values.to(device, dtype=torch.float32)
            tokens = reader.generate(pixels, do_sample=False, num_beams=1,
                                     max_new_tokens=cfg['max_new_tokens'])
        return processor.batch_decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def update(name, target, prediction):
        value = totals[name]
        value['samples'] += 1; value['characters'] += len(target)
        value['char_edits'] += levenshtein(prediction, target)
        # Same word-crop error definition as MAIN's independent OCR evaluator.
        value['wrong_words'] += prediction != target

    with IAMCorpus(cfg['dataset']) as corpus:
        rows, unsupported = selected_rows(corpus.texts, common, cfg['unsupported_policy'], cfg['limit'])
        rows_hash = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
        predictions = output/'predictions.jsonl.partial'
        with predictions.open('w', encoding='utf-8') as stream:
            for start in range(0, len(rows), cfg['batch_size']):
                indices = rows[start:start+cfg['batch_size']]
                real, generated, targets = [], [], []
                for i in indices:
                    text = corpus.texts[i]; image = corpus.image(i)
                    if cfg.get('source_polarity', 'white') == 'black': image = 255-image
                    fake = adapter.generate(image, text, text, cfg['seed']+i)
                    if fake.shape[0] != corpus.height:
                        if not cfg['convert_to_dataset_height']:
                            raise ValueError('Native/dataset height differs; enable explicit conversion or use a native cohort')
                        width = max(1, round(fake.shape[1]*corpus.height/fake.shape[0]))
                        fake = np.asarray(Image.fromarray(fake).resize((width, corpus.height), Image.Resampling.BILINEAR))
                    real.append(Image.fromarray(image).convert('RGB'))
                    generated.append(Image.fromarray(fake).convert('RGB')); targets.append(text)
                fake_predictions = recognize(generated)
                real_predictions = recognize(real) if cfg['real_control'] else [None]*len(indices)
                for i, target, predicted, original in zip(indices, targets, fake_predictions, real_predictions):
                    update('generated', target, predicted)
                    if original is not None: update('real', target, original)
                    stream.write(json.dumps(dict(index=i, target=target, generated=predicted, real=original), ensure_ascii=False)+'\n')
                stream.flush()
                if start % (cfg['batch_size']*50) == 0:
                    print(f'{model_name}: {start+len(indices)}/{len(rows)}', flush=True)
                    if cfg['wandb_mode'] != 'disabled':
                        import wandb
                        wandb.log({'samples': start+len(indices)})
        for value in totals.values():
            value['cer'] = value['char_edits']/value['characters']
            value['wer'] = value['wrong_words']/value['samples']
            value['cer_percent'] = 100*value['cer']; value['wer_percent'] = 100*value['wer']
        result = dict(model=model_name, metrics=totals, dataset_sha256=sha256_file(cfg['dataset']),
                      row_indices_sha256=rows_hash, population=len(corpus), evaluated=len(rows),
                      unsupported_rows=len(unsupported), unsupported_policy=cfg['unsupported_policy'],
                      limited_diagnostic=cfg['limit'] is not None, protocol='same-transcript-guided-IAM',
                      ocr_model=cfg['ocr_model'], ocr_revision=cfg['ocr_revision'],
                      max_new_tokens=cfg['max_new_tokens'], precision='float32',
                      native_height=profile['native_height'], evaluation_height=corpus.height,
                      conversion='bilinear' if profile['native_height'] != corpus.height else 'none',
                      generator_provenance=adapter.metadata)
        os.replace(predictions, output/'predictions.jsonl')
        atomic_json(output/'metrics.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--model', required=True)
    parser.add_argument('--status', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    output = Path(cfg['output_root'])/args.model; output.mkdir(parents=True, exist_ok=True)
    status = dict(branch=args.model, job=args.model+'-OCR_EVAL', pid=os.getpid(), logdir=str(output.resolve()))
    atomic_json(args.status, dict(status='running', **status))
    tracking = None
    try:
        if cfg['batch_size'] not in range(1,5): raise ValueError('batch_size must be1..4')
        if cfg.get('source_polarity', 'white') not in ('white', 'black'):
            raise ValueError('source_polarity must be white or black')
        if cfg['wandb_mode'] != 'disabled':
            import wandb
            tracking = wandb.init(project=cfg['wandb_project'], name=args.model+'-ocr-eval',
                                  mode=cfg['wandb_mode'], dir=str(output), config=cfg)
        result = run(cfg, args.model, output)
        if tracking:
            tracking.log({name+'/'+metric: values[metric] for name, values in result['metrics'].items()
                          for metric in ('cer_percent','wer_percent')})
        print('RESULTS', json.dumps(result['metrics']), flush=True)
        final = dict(status='completed', results_path=str(output/'metrics.json'), **status)
    except BaseException as error:
        final = dict(status='failed', error=f'{type(error).__name__}: {error}', **status)
        raise
    finally:
        try:
            if tracking: tracking.finish()
        finally:
            atomic_json(args.status, final)


if __name__ == '__main__': main()
