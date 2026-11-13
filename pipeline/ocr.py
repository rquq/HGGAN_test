"""Full-parameter pinned TrOCR benchmark with native HF checkpoint resume.

One process per device/job. Real validation alone selects (CER, WER); the full
literal real test is streamed only from that selected checkpoint. No GAN import
or generator checkpoint is needed by the portable synthetic dataset gate.
"""
import csv
from functools import partial
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from PIL import Image
import torch
from transformers import (GenerationConfig, Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          TrainerCallback, TrainerState, TrOCRProcessor,
                          VisionEncoderDecoderModel, set_seed)
from transformers.trainer_callback import ExportableState

from .common import atomic_json, bind_identity, read_json, sha256_file
from .config import DEFAULTS
from .data import IAMCorpus
from .manifest import prepare
from .metrics import RecognitionMetrics
from .tracking import tracking_run


def settings(cfg):
    result = dict(DEFAULTS['ocr'], **cfg.get('ocr', {}))
    if (result['model_name'] != DEFAULTS['ocr']['model_name'] or
            result['revision'] != DEFAULTS['ocr']['revision']):
        raise ValueError('only the pinned TrOCR-small-stage1 model is supported')
    for key in ('max_steps', 'micro_batch_size', 'gradient_accumulation_steps',
                'eval_steps', 'save_steps', 'max_new_tokens'):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError(f'ocr.{key} must be a positive integer')
    if result['save_steps'] != result['eval_steps']:
        raise ValueError('evaluation and save intervals must be aligned')
    if result['num_beams'] != 1 or result['fp16'] or result['bf16']:
        raise ValueError('this benchmark requires greedy generation and FP32')
    if not 0 <= result['warmup_ratio'] <= 1:
        raise ValueError('invalid warmup fraction')
    return result


def load_processor(options):
    return TrOCRProcessor.from_pretrained(options['model_name'], revision=options['revision'])


def load_model(options):
    return VisionEncoderDecoderModel.from_pretrained(
        options['model_name'], revision=options['revision'], dtype=torch.float32)


def _load_checkpoint_model(path):
    """Apply HF's supported legacy-key conversion, but reject partial restores.

    Transformers 5.9 save_pretrained writes legacy DeiT encoder names; Trainer's
    raw load_state_dict path does not convert them. from_pretrained does, and
    also restores tied decoder weights omitted from safetensors serialization.
    """
    model, info = VisionEncoderDecoderModel.from_pretrained(
        path, dtype=torch.float32, local_files_only=True, output_loading_info=True)
    errors = {key: info[key] for key in ('missing_keys', 'unexpected_keys',
                                        'mismatched_keys', 'error_msgs') if info.get(key)}
    if errors:
        raise ValueError(f'incomplete OCR weight restore: {errors}')
    return model


class OCRTrainer(Seq2SeqTrainer):
    """Pinned weight-loading bridge; native Trainer owns all training state."""
    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        target = self.model if model is None else model
        restored = _load_checkpoint_model(resume_from_checkpoint)
        # Copy into existing parameters: optimizer references must not change.
        target.load_state_dict(restored.state_dict(), strict=True)

    def _load_best_model(self):
        # Native best reload has the same raw-key bug in this pinned release.
        self._load_from_checkpoint(self.state.best_model_checkpoint)


class OCRDataset(torch.utils.data.Dataset):
    """Keep images on disk and tokenize literal text once, never truncate."""
    def __init__(self, paths, processor, max_label_length=512):
        self.corpora, self.rows, self.texts, self.labels = [], [], [], []
        self.processor = processor
        try:
            for path in paths:
                corpus = IAMCorpus(path)
                self.corpora.append(corpus)
                for index, text in enumerate(corpus.texts):
                    ids = processor.tokenizer(text, truncation=False).input_ids
                    if len(ids) > max_label_length:
                        raise ValueError(f'label length {len(ids)} exceeds decoder capacity {max_label_length}')
                    self.rows.append((len(self.corpora)-1, index))
                    self.texts.append(text)
                    self.labels.append(torch.tensor(ids, dtype=torch.long))
            if not self.rows:
                raise ValueError('OCR dataset must not be empty')
        except BaseException:
            self.close()
            raise

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        source, row = self.rows[index]
        image = Image.fromarray(self.corpora[source].image(row)).convert('RGB')
        pixels = self.processor(images=image, return_tensors='pt').pixel_values[0]
        return {'pixel_values': pixels, 'labels': self.labels[index]}

    def close(self):
        for corpus in self.corpora:
            corpus.close()
        self.corpora = []


def collate(items, pad_token_id):
    labels = torch.nn.utils.rnn.pad_sequence([x['labels'] for x in items],
                                             batch_first=True, padding_value=pad_token_id)
    labels[labels == pad_token_id] = -100
    return dict(pixel_values=torch.stack([x['pixel_values'] for x in items]), labels=labels)


def length_limit_hit(ids, max_new_tokens, eos_token_id=2, pad_token_id=1):
    generated = [int(x) for x in ids[1:] if int(x) >= 0 and int(x) != pad_token_id]
    return len(generated) >= max_new_tokens and eos_token_id not in generated[:max_new_tokens]


class StreamingValidation:
    def __init__(self, processor, texts, max_new_tokens):
        self.processor, self.texts, self.max_new_tokens = processor, texts, max_new_tokens
        self.reset()

    def reset(self):
        self.metrics, self.cursor, self.limit_hits = RecognitionMetrics(), 0, 0

    def __call__(self, prediction, compute_result=False):
        ids = prediction.predictions
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().numpy()
        ids = np.where(ids < 0, self.processor.tokenizer.pad_token_id, ids)
        predictions = self.processor.batch_decode(ids, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
        for tokens, text in zip(ids, predictions):
            self.metrics.update(self.texts[self.cursor], text)
            self.cursor += 1
            self.limit_hits += length_limit_hit(tokens, self.max_new_tokens)
        if not compute_result:
            return {}
        if self.cursor != len(self.texts):
            raise ValueError('validation prediction order/count mismatch')
        result = dict(self.metrics.compute(), length_limit_hits=self.limit_hits)
        self.reset()
        return result


class BestHistoryCallback(TrainerCallback, ExportableState):
    """Persist tuple comparator before native checkpoint state serialization."""
    def __init__(self):
        self.best_pair = None

    def state(self):
        return {'args': {}, 'attributes': {'best_pair': self.best_pair}}

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        pair = (float(metrics['eval_cer']), float(metrics['eval_wer']))
        if not all(np.isfinite(pair)):
            raise ValueError('nonfinite validation selection metrics')
        if self.best_pair is None or pair < tuple(self.best_pair):
            self.best_pair = list(pair)
            state.best_metric = pair[0]
            state.best_global_step = state.global_step
        control.should_save = True


class BudgetExpired(Exception):
    """Internal cooperative validation boundary, never a partial result."""


class BudgetCallback(TrainerCallback):
    """Never shorten max_steps/schedule; stop only after a full optimizer update."""
    def __init__(self, deadline=None):
        self.deadline = deadline
        self.paused = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.paused = True
            control.should_training_stop = control.should_save = True
            control.should_evaluate = False
        elif state.global_step >= args.max_steps:
            control.should_evaluate = control.should_save = True

    def on_prediction_step(self, args, state, control, **kwargs):
        if self.deadline is not None and time.monotonic() >= self.deadline:
            self.paused = True
            raise BudgetExpired()


class ReceiptCallback(TrainerCallback):
    def __init__(self, directory, tracking):
        self.directory, self.tracking = Path(directory), tracking

    def on_log(self, args, state, control, logs, **kwargs):
        self.tracking.log(logs, step=state.global_step)

    def on_save(self, args, state, control, **kwargs):
        _receipts(self.directory, state)


def _atomic_csv(path, rows):
    keys = sorted({key for row in rows for key in row})
    fd, temporary = tempfile.mkstemp(prefix='.history-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _pointer(directory, name, checkpoint):
    target = directory / name
    if target.exists() and not target.is_symlink():
        # A download may materialize aliases. Preserve only a verified duplicate
        # as a recoverable backup; never remove an unrelated user's directory.
        if not any(_same_checkpoint_tree(target, candidate)
                   for candidate in directory.glob('checkpoint-*') if candidate.is_dir()):
            raise ValueError(f'conflicting checkpoint alias: {target}')
        backup = Path(tempfile.mkdtemp(prefix='.restored-alias-', dir=directory))
        os.replace(target, backup / name)
    temporary = directory / ('.' + name)
    if temporary.is_symlink():
        temporary.unlink()
    temporary.symlink_to(Path(checkpoint).name, target_is_directory=True)
    os.replace(temporary, target)


def _receipts(directory, state):
    last = directory / f'checkpoint-{state.global_step}'
    _validate_checkpoint(last)
    _pointer(directory, 'last', last)
    if state.best_model_checkpoint:
        best = directory / Path(state.best_model_checkpoint).name
        _validate_checkpoint(best)
        _pointer(directory, 'best', best)
        atomic_json(directory / 'best.json', dict(checkpoint=best.name,
                    global_step=state.best_global_step, cer=state.best_metric,
                    pair=state.stateful_callbacks.get('BestHistoryCallback', {})
                    .get('attributes', {}).get('best_pair')))
    # Checkpoint history is authoritative. Deduplicate repeated kind/step records.
    unique = {}
    for row in read_json(last / 'trainer_state.json')['log_history']:
        kind = 'validation' if 'eval_cer' in row else ('train' if 'loss' in row else 'runtime')
        unique[(kind, row.get('step'))] = dict(kind=kind, **row)
    _atomic_csv(directory / 'history.csv', list(unique.values()))


def _validate_checkpoint(path):
    required = ['trainer_state.json', 'optimizer.pt', 'scheduler.pt', 'rng_state.pth', 'config.json']
    if not path.is_dir() or any(not (path / name).is_file() for name in required):
        raise ValueError(f'incomplete resumable checkpoint: {path}')
    if not any((path / name).is_file() for name in ('model.safetensors', 'pytorch_model.bin',
                                                  'model.safetensors.index.json')):
        raise ValueError(f'checkpoint is missing model weights: {path}')
    state = read_json(path / 'trainer_state.json')
    if int(path.name.split('-')[-1]) != state['global_step']:
        raise ValueError('checkpoint step does not match its trainer state')
    return state


def _same_checkpoint_tree(alias, canonical):
    def signature(root):
        if not root.is_dir() or root.is_symlink():
            raise ValueError('checkpoint alias must be a directory')
        result = {}
        for path in root.rglob('*'):
            if path.is_symlink():
                raise ValueError('unsafe link inside materialized checkpoint')
            result[str(path.relative_to(root))] = sha256_file(path) if path.is_file() else None
        return result
    return signature(alias) == signature(canonical)


def _checkpoint_number(name):
    if not isinstance(name,str) or not name.startswith('checkpoint-'):
        raise ValueError('invalid canonical checkpoint name')
    value = name.removeprefix('checkpoint-')
    if not value.isdigit() or str(int(value)) != value:
        raise ValueError('invalid canonical checkpoint step')
    return int(value)


def _check_alias_representation(alias, canonical):
    limit = _checkpoint_number(canonical.name)
    if alias.is_symlink():
        recorded = os.readlink(alias)
        step = _checkpoint_number(recorded)
        if step > limit:
            raise ValueError(f'unsafe/conflicting checkpoint alias: {alias}')
        prior = canonical.parent / recorded
        if prior.is_symlink():
            raise ValueError('canonical checkpoint must not be a symlink')
        if prior.exists():
            _validate_checkpoint(prior)
        # A safe older relative link may dangle after native checkpoint rotation.
    elif alias.exists():
        if not alias.is_dir():
            raise ValueError(f'conflicting checkpoint alias: {alias}')
        try:
            state = read_json(alias/'trainer_state.json')
            prior = canonical.parent / f'checkpoint-{state["global_step"]}'
            if _checkpoint_number(prior.name) > limit or not _same_checkpoint_tree(alias, prior):
                raise ValueError(f'incomplete/conflicting checkpoint alias: {alias}')
        except (OSError,KeyError,TypeError) as exc:
            raise ValueError(f'incomplete/conflicting checkpoint alias: {alias}') from exc


def _check_best_receipt(path, selected):
    if path.is_symlink():
        raise ValueError('best receipt must not be a symlink')
    if not path.exists():
        return None
    receipt = read_json(path)
    recorded = receipt.get('checkpoint')
    step = _checkpoint_number(recorded)
    if step != receipt.get('global_step') or (selected and step > _checkpoint_number(selected)):
        raise ValueError('best checkpoint receipt conflicts with native selection')
    return recorded


def _checkpoint_aliases(directory):
    """Canonical native state is authoritative; aliases/receipts may lag a save."""
    checkpoints = [p for p in directory.glob('checkpoint-*') if p.is_dir() and not p.is_symlink()]
    if not checkpoints:
        if any((directory/name).exists() or (directory/name).is_symlink() for name in ('best','last','best.json')):
            raise ValueError('checkpoint aliases have no canonical checkpoint-N save')
        return {}
    states = {p.name:_validate_checkpoint(p) for p in checkpoints}
    latest = max(checkpoints, key=lambda p: states[p.name]['global_step'])
    plan = {'last':latest.name}
    state_best = states[latest.name].get('best_model_checkpoint')
    selected = Path(state_best).name if state_best else None
    recorded = _check_best_receipt(directory/'best.json',selected)
    if recorded:
        if not selected:
            if 'best_model_checkpoint' in states[latest.name]:
                raise ValueError('best receipt conflicts with native no-selection state')
            selected = recorded  # Legacy canonical state without this native key.
    if selected:
        if selected not in states:
            raise ValueError('selected canonical best checkpoint is missing')
        plan['best'] = selected
    for name in ('best','last'):
        alias = directory/name
        if name not in plan and (alias.exists() or alias.is_symlink()):
            raise ValueError('best alias has no authoritative validation selection')
        if name in plan:
            _check_alias_representation(alias, directory/plan[name])
    return plan


def _normalize_checkpoint_aliases(directory, plan=None):
    plan = _checkpoint_aliases(directory) if plan is None else plan
    # Validate every alias before changing either. Only writable output calls us.
    for name, canonical in plan.items():
        _check_alias_representation(directory/name, directory/canonical)
    for name, canonical in plan.items():
        _pointer(directory,name,directory/canonical)
    if 'best' in plan:
        state = _validate_checkpoint(directory/plan['last'])
        atomic_json(directory/'best.json',dict(checkpoint=plan['best'],
            global_step=_checkpoint_number(plan['best']),cer=state.get('best_metric'),
            pair=state.get('stateful_callbacks',{}).get('BestHistoryCallback',{})
                .get('attributes',{}).get('best_pair')))
    return plan


def _latest(directory):
    _normalize_checkpoint_aliases(directory)
    checkpoints = [p for p in directory.glob('checkpoint-*') if p.is_dir()]
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: int(p.name.split('-')[-1]))
    state = _validate_checkpoint(latest)
    # Native state stores an absolute best path. Rebind only the path (content
    # identity was checked first) when restoring a writable relocated run.
    if state.get('best_model_checkpoint'):
        best = directory / Path(state['best_model_checkpoint']).name
        _validate_checkpoint(best)
        if state['best_model_checkpoint'] != str(best):
            state['best_model_checkpoint'] = str(best)
            atomic_json(latest / 'trainer_state.json', state)
    return latest


def _job(cfg, arm, seed):
    if arm not in {'real_only', 'spis_gan', 'fw_gan', 'higanplus'}:
        raise ValueError(f'unknown OCR arm: {arm}')
    options = settings(cfg)
    prepared = prepare(cfg)
    train_paths = [prepared / 'real_train.hdf5']
    payload = dict(arm=arm, seed=int(seed), ocr={k:v for k,v in options.items() if k != 'seeds'},
                   prepared=read_json(prepared / 'identity.json')['sha256'],
                   real_train=sha256_file(train_paths[0]),
                   validation=sha256_file(prepared / 'real_val.hdf5'),
                   test=sha256_file(cfg['dataset']['test']))
    if arm != 'real_only':
        from .generation import verify_dataset
        metadata = verify_dataset(cfg, arm, verify_model=False)
        synthetic = Path(cfg['output_dir']) / 'datasets' / arm / 'synthetic.hdf5'
        payload['synthetic'] = dict(identity=metadata['identity_sha256'], sha256=sha256_file(synthetic))
        train_paths.append(synthetic)
    directory = Path(cfg['output_dir']) / 'ocr' / arm / f'seed_{int(seed)}'
    identity = bind_identity(directory, payload)
    return options, prepared, train_paths, directory, identity


def _generation(model, processor, options):
    pad, eos = processor.tokenizer.pad_token_id, processor.tokenizer.eos_token_id
    model.config.pad_token_id, model.config.decoder_start_token_id = pad, eos
    model.accepts_loss_kwargs = False  # VED CE ignores num_items_in_batch.
    model.generation_config = GenerationConfig(decoder_start_token_id=eos, eos_token_id=eos,
        pad_token_id=pad, max_new_tokens=options['max_new_tokens'], num_beams=1,
        do_sample=False, return_dict_in_generate=False, output_scores=False)


def _arguments(directory, options, seed, device):
    if device != 'cpu' and not device.startswith('cuda'):
        raise ValueError('OCR device must be cpu or cuda')
    if device.startswith('cuda'):
        if not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable')
        torch.cuda.set_device(torch.device(device))
    return Seq2SeqTrainingArguments(output_dir=str(directory), max_steps=options['max_steps'],
        per_device_train_batch_size=options['micro_batch_size'],
        per_device_eval_batch_size=options['micro_batch_size'],
        gradient_accumulation_steps=options['gradient_accumulation_steps'],
        learning_rate=options['learning_rate'], weight_decay=options['weight_decay'],
        max_grad_norm=options['max_grad_norm'], warmup_steps=options['warmup_ratio'],
        lr_scheduler_type='linear', optim='adamw_torch', fp16=False, bf16=False,
        eval_strategy='steps', save_strategy='steps', eval_steps=options['eval_steps'],
        save_steps=options['save_steps'], save_total_limit=2, save_only_model=False,
        load_best_model_at_end=False, metric_for_best_model='cer', greater_is_better=False,
        restore_callback_states_from_checkpoint=True, ignore_data_skip=False,
        predict_with_generate=True, batch_eval_metrics=True, eval_accumulation_steps=1,
        report_to=[], seed=int(seed), data_seed=int(seed), use_cpu=device=='cpu',
        dataloader_num_workers=0, dataloader_pin_memory=device!='cpu',
        logging_steps=1, remove_unused_columns=False, disable_tqdm=True)


def _status(directory, identity, step, status, **extra):
    result = dict(status=status, identity_sha256=identity, global_step=int(step), **extra)
    atomic_json(directory / 'status.json', result)
    return result


def _committed_step(directory, fallback):
    """Read-only error-path scan; incomplete newest saves do not hide older ones."""
    try:
        steps = []
        for path in directory.glob('checkpoint-*'):
            try:
                steps.append(int(_validate_checkpoint(path)['global_step']))
            except (OSError,ValueError,KeyError,TypeError):
                continue
        return max(steps, default=fallback)
    except Exception:
        return fallback


def train(cfg, arm, seed, device='cpu', time_budget_seconds=None):
    """Train/resume to the ORIGINAL horizon, then test the validation-only best."""
    deadline = None if time_budget_seconds is None else time.monotonic()+time_budget_seconds
    options, prepared, paths, directory, identity = _job(cfg, arm, seed)
    latest = _latest(directory)
    step = _validate_checkpoint(latest)['global_step'] if latest else 0
    start_step = step
    if step > options['max_steps']:
        raise ValueError('checkpoint exceeds configured optimizer budget')
    prior = directory / 'status.json'
    if prior.exists() and read_json(prior).get('status') == 'complete':
        _verify_final(directory, identity)
        return read_json(prior)
    data = []
    try:
        with tracking_run(cfg, directory, 'train', arm, seed, {'identity':identity}) as tracking:
            set_seed(int(seed))
            processor = load_processor(options)
            model = load_model(options)
            print(f'OCR {arm} seed={seed}: FP32 device={device} '
                  f'parameters={sum(p.numel() for p in model.parameters())} '
                  f'micro_batch={options["micro_batch_size"]} '
                  f'accumulation={options["gradient_accumulation_steps"]}',flush=True)
            _generation(model, processor, options)
            capacity = model.config.decoder.max_position_embeddings
            training = OCRDataset(paths, processor, capacity)
            data.append(training)
            validation = OCRDataset([prepared/'real_val.hdf5'], processor, capacity)
            data.append(validation)
            budget = BudgetCallback(deadline)
            trainer = OCRTrainer(model=model, args=_arguments(directory, options, seed, device),
                processing_class=processor, train_dataset=training, eval_dataset=validation,
                data_collator=partial(collate,pad_token_id=processor.tokenizer.pad_token_id),
                compute_metrics=StreamingValidation(processor,validation.texts,options['max_new_tokens']),
                callbacks=[BestHistoryCallback(),budget,ReceiptCallback(directory,tracking)])
            if step < options['max_steps']:
                try:
                    trainer.train(resume_from_checkpoint=str(latest) if latest else None)
                except BudgetExpired:
                    # Evaluation began after a complete optimizer update. Use
                    # HF's checkpoint writer, including optimizer/scheduler/RNG,
                    # but do not publish partial validation or invent a best.
                    trainer._save_checkpoint(trainer.model, None)
                    _receipts(directory,trainer.state)
                    return _status(directory,identity,trainer.state.global_step,'paused',phase='validation')
                step = trainer.state.global_step
            else:
                # A completed optimizer horizon must never take another update.
                trainer._load_from_checkpoint(str(latest))
                trainer.state = TrainerState.load_from_json(latest/'trainer_state.json')
                trainer._load_callback_state()
            latest = _latest(directory)
            step = _validate_checkpoint(latest)['global_step']
            _receipts(directory, trainer.state)
            if budget.paused or step < options['max_steps'] or (deadline is not None and time.monotonic() >= deadline):
                return _status(directory,identity,step,'paused',phase='training' if step<options['max_steps'] else 'validation')
            # A budget pause can suppress evaluation at the last update. Validate
            # those exact saved weights on continuation without touching optimizer.
            eval_steps = {r['step'] for r in trainer.state.log_history if 'eval_cer' in r}
            if step not in eval_steps:
                try:
                    trainer.evaluate()
                except BudgetExpired:
                    return _status(directory,identity,step,'paused',phase='validation')
                trainer._determine_best_metric(trainer.state.log_history[-1], None)
                if trainer.state.best_global_step == step:
                    trainer.state.best_model_checkpoint = str(latest)
                for cb in trainer.callback_handler.callbacks:
                    if isinstance(cb, ExportableState):
                        trainer.state.stateful_callbacks[type(cb).__name__] = cb.state()
                trainer.state.save_to_json(latest/'trainer_state.json')
                _receipts(directory,trainer.state)
            if deadline is not None and time.monotonic() >= deadline:
                return _status(directory,identity,step,'paused',phase='test')
            # Release optimizer/training model before loading the selected model.
            del trainer, model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return _evaluate(cfg,arm,seed,device,options,directory,identity,deadline,tracking)
    except KeyboardInterrupt:
        try:
            _status(directory,identity,_committed_step(directory,start_step),'paused',phase='interrupted')
        except Exception:
            pass  # Preserve the original interruption even if storage is unavailable.
        raise
    except BaseException as exc:
        try:
            _status(directory,identity,_committed_step(directory,start_step),'failed',error=type(exc).__name__)
        except Exception:
            pass  # Reporting must never replace the original training exception.
        raise
    finally:
        for dataset in data:
            dataset.close()


def _verify_final(directory, identity):
    receipt = read_json(directory/'test_metrics.json')
    if receipt['identity_sha256'] != identity or sha256_file(directory/'test_predictions.jsonl') != receipt['predictions_sha256']:
        raise ValueError('final test artifact identity/hash mismatch')
    return receipt


def _evaluate(cfg,arm,seed,device,options,directory,identity,deadline,tracking):
    best = directory/'best'
    if not best.exists():
        raise ValueError('no validation-selected best checkpoint exists')
    state = _validate_checkpoint(best.resolve())
    if (directory/'test_metrics.json').exists():
        metrics = _verify_final(directory,identity)
        return _status(directory,identity,options['max_steps'],'complete',metrics=metrics)
    processor = load_processor(options)
    model = _load_checkpoint_model(best.resolve())
    _generation(model,processor,options)
    model.to(device).eval()
    dataset = OCRDataset([cfg['dataset']['test']],processor,model.config.decoder.max_position_embeddings)
    fd, temporary = tempfile.mkstemp(prefix='.test-predictions-',dir=directory)
    metrics, limit_hits = RecognitionMetrics(), 0
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream, torch.inference_mode():
            for start in range(0,len(dataset),options['micro_batch_size']):
                if deadline is not None and time.monotonic() >= deadline:
                    return _status(directory,identity,options['max_steps'],'paused',phase='test')
                end = min(start+options['micro_batch_size'],len(dataset))
                batch = collate([dataset[i] for i in range(start,end)],processor.tokenizer.pad_token_id)
                tokens = model.generate(batch['pixel_values'].to(device)).cpu().numpy()
                predictions = processor.batch_decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
                if len(tokens) != end-start or len(predictions) != end-start:
                    raise ValueError('test prediction batch count mismatch')
                for index,ids,prediction in zip(range(start,end),tokens,predictions):
                    target=dataset.texts[index]
                    metrics.update(target,prediction)
                    hit=length_limit_hit(ids,options['max_new_tokens'])
                    limit_hits+=hit
                    stream.write(json.dumps(dict(row=index,target=target,prediction=prediction,
                        length_limit_hit=hit),ensure_ascii=False)+'\n')
            stream.flush()
            os.fsync(stream.fileno())
        if metrics.samples != len(dataset):
            raise ValueError('full test prediction count mismatch')
        # Full supplied test completed. No partial metrics/predictions published.
        prediction_path=directory/'test_predictions.jsonl'
        os.replace(temporary,prediction_path)
        result=dict(metrics.compute(),identity_sha256=identity,selection='validation_cer_wer',
                    checkpoint=best.resolve().name,checkpoint_step=state['global_step'],
                    length_limit_hits=limit_hits,predictions_sha256=sha256_file(prediction_path))
        atomic_json(directory/'test_metrics.json',result)
        tracking.log({'test/'+k:v for k,v in result.items() if isinstance(v,(int,float))})
        return _status(directory,identity,options['max_steps'],'complete',metrics=result)
    finally:
        dataset.close()
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate(cfg, arm, seed, device='cpu'):
    """Independent full-test evaluation of an already validation-selected run."""
    options,_,_,directory,identity=_job(cfg,arm,seed)
    latest=_latest(directory)
    if latest is None or _validate_checkpoint(latest)['global_step'] != options['max_steps']:
        raise ValueError('final test requires the full configured optimizer budget')
    state=_validate_checkpoint(latest)
    if options['max_steps'] not in {r['step'] for r in state['log_history'] if 'eval_cer' in r}:
        raise ValueError('final validation is pending; resume train before test evaluation')
    try:
        with tracking_run(cfg,directory,'evaluate',arm,seed,{'identity':identity}) as tracking:
            return _evaluate(cfg,arm,seed,device,options,directory,identity,None,tracking)
    except KeyboardInterrupt:
        _status(directory,identity,options['max_steps'],'paused',phase='test')
        raise
    except BaseException as exc:
        _status(directory,identity,options['max_steps'],'failed',phase='test',error=type(exc).__name__)
        raise
