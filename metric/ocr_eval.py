"""Bounded, independent reference-transfer OCR evaluation (CER and WER only)."""
import csv
import gc
import hashlib
import json
import random
import time
from contextlib import contextmanager
from itertools import zip_longest
from pathlib import Path

import h5py
import numpy as np
import torch
from distance import levenshtein
from PIL import Image, ImageDraw

OCR_MODEL = 'microsoft/trocr-small-handwritten'
OCR_REVISION = 'b4648cfa171985a6745f37ddd637e98c0da958ac'
METRIC_VERSION = 'literal-transcripts-v1'
PREPROCESS_VERSION = 'canonical-white-paper-v1'


def file_fingerprint(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''): digest.update(chunk)
    return digest.hexdigest()


def recognition_scores(targets, predictions):
    if not targets or len(targets) != len(predictions) or any(not t for t in targets):
        raise ValueError('nonempty paired transcripts with nonempty targets are required')
    return {'cer': sum(levenshtein(p, t) for t, p in zip(targets, predictions)) / sum(map(len, targets)),
            'wer': sum(t != p for t, p in zip(targets, predictions)) / len(targets)}


@contextmanager
def evaluation_context(models):
    flags = {m: m.training for model in models for m in model.modules()}
    state = torch.get_rng_state(), np.random.get_state(), random.getstate()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        for model in models: model.eval()
        with torch.inference_mode(): yield
    finally:
        for module, flag in flags.items(): module.training = flag
        torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])
        if cuda_state is not None: torch.cuda.set_rng_state_all(cuda_state)


class WordCorpus:
    """Keep text metadata in RAM; read packed image slices on demand."""
    def __init__(self, path, source_polarity='white'):
        self.path = Path(path).resolve()
        self.source_polarity = source_polarity
        with h5py.File(self.path) as f:
            lbs, starts, lens = f['lbs'][:], f['lb_seek_idxs'][:], f['lb_lens'][:]
            self.texts = [''.join(map(chr, lbs[s:s+n])) for s, n in zip(starts, lens)]
            self.writers = f['wids'][:].astype(int).tolist()
            self.image_starts, self.widths = f['img_seek_idxs'][:], f['img_lens'][:]
            self.height = int(f['imgs'].shape[0])
        payload = json.dumps([self.texts, self.writers], ensure_ascii=False).encode()
        self.fingerprint = hashlib.sha256(payload).hexdigest()
        self.image_fingerprint = file_fingerprint(self.path)

    def metadata(self): return {'texts': self.texts, 'writers': self.writers}

    def image(self, index):
        s, w = int(self.image_starts[index]), int(self.widths[index])
        with h5py.File(self.path) as f: a = f['imgs'][:, s:s+w].astype(np.float32)
        # Canonical model convention: -1 paper, +1 ink.
        if self.source_polarity == 'white': a = 255 - a
        return torch.from_numpy(a / 127.5 - 1).unsqueeze(0)


def unexposed_words(candidates, exposure):
    """Exclude whole-word and substring exposure from the training crop sampler."""
    trie = {}
    for word in candidates:
        node = trie
        for char in word: node = node.setdefault(char, {})
        node[None] = word
    found = set()
    for word in exposure:
        for start in range(len(word)):
            node = trie
            for char in word[start:]:
                if char not in node: break
                node = node[char]
                if None in node: found.add(node[None])
        if len(found) == len(candidates): break
    return set(candidates) - found


def make_cases(train, test, lexicon, alphabet, max_word_len=20, per_cell=64, pairs=32, seed=123456, writer_splits_disjoint=False):
    if per_cell < 1 or pairs < 0: raise ValueError('per_cell >= 1 and pairs >= 0 required')
    rng = np.random.default_rng(seed)
    allowed = lambda t: bool(t) and len(t) <= max_word_len and set(t) <= alphabet
    real_words = set(train['texts'])
    exposed = real_words | set(lexicon)
    # Training may capitalize or uppercase sampled lexicon entries.
    exposed |= {w.capitalize() for w in real_words | set(lexicon)} | {w.upper() for w in real_words | set(lexicon)}
    candidates = {w for w in set(test['texts']) - exposed if allowed(w)}
    pools = {'IV': sorted(w for w in real_words if allowed(w)),
             'OOV': sorted(unexposed_words(candidates, exposed))}
    cases = []; counts = {}; notes = []
    if not writer_splits_disjoint:
        notes.append('Writer identity across files is unverified; groups denote reference split, not seen/unseen writers.')
    for vocab in ['IV', 'OOV']:
        for style, split, data in [('S','train',train),('U','test',test)]:
            scenario = f'{vocab}-{style}' if writer_splits_disjoint else f'{vocab}-{split}ref'
            by_writer = {}
            for i, (text, writer) in enumerate(zip(data['texts'], data['writers'])):
                if not allowed(text): continue
                by_writer.setdefault(writer, []).append(i)
            writers = sorted(by_writer); rng.shuffle(writers)
            if not writers or not pools[vocab]:
                counts[scenario] = 0; notes.append(f'{scenario}: no eligible references or unexposed target words'); continue
            for j in range(per_cell):
                writer = writers[j % len(writers)]
                index = int(rng.choice(by_writer[writer]))
                candidates = [w for w in pools[vocab] if w != data['texts'][index]]
                if not candidates: continue
                cases.append({'scenario': scenario, 'split': split, 'index': index,
                              'writer': writer, 'reference': data['texts'][index], 'target': str(rng.choice(candidates))})
            counts[scenario] = sum(c['scenario'] == scenario for c in cases)
    short_by_writer, long_by_writer = {}, {}
    for i, (text, writer) in enumerate(zip(test['texts'], test['writers'])):
        if not allowed(text): continue
        if len(text) <= 2: short_by_writer.setdefault(writer, []).append(i)
        if len(text) >= 6: long_by_writer.setdefault(writer, []).append(i)
    writers = sorted(short_by_writer.keys() & long_by_writer.keys()); rng.shuffle(writers)
    for pair, writer in enumerate(writers[:pairs]):
        short = int(rng.choice(short_by_writer[writer])); long = int(rng.choice(long_by_writer[writer]))
        targets = [w for w in pools['IV'] if w not in (test['texts'][short], test['texts'][long])]
        if not targets: continue
        target = str(rng.choice(targets))
        for scenario, index in [('short',short), ('long',long)]:
            cases.append({'scenario':scenario,'split':'test','index':index,'writer':writer,
                          'reference':test['texts'][index],'target':target,'pair':pair})
    for label in ['short','long']: counts[label] = sum(c['scenario']==label for c in cases)
    if counts['short'] < pairs: notes.append(f'only {counts["short"]} eligible short/full writer pairs')
    return {'version':1,'seed':seed,'cases':cases,'counts':counts,'notes':notes}


def transfer_manifest(train, test, lexicon, converter, path, per_cell=64, pairs=32, max_word_len=20, seed=123456, writer_splits_disjoint=False):
    identity = {'protocol_version':2,'train':train.fingerprint,'test':test.fingerprint,
                'lexicon':hashlib.sha256('\n'.join(sorted(lexicon)).encode()).hexdigest(),
                'alphabet':converter.alphabet,'per_cell':per_cell,'pairs':pairs,'max_word_len':max_word_len,'seed':seed,
                'writer_splits_disjoint':writer_splits_disjoint}
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text())
        if manifest.get('identity') != identity:
            raise ValueError(f'manifest identity differs: {path}; choose a new output path for changed data/protocol')
        return manifest
    manifest = make_cases(train.metadata(), test.metadata(), set(lexicon), set(converter.alphabet)-{converter.alphabet[0]}, max_word_len, per_cell, pairs, seed, writer_splits_disjoint)
    manifest['identity'] = identity
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    return manifest


def canonical_pil(image):
    a = ((1-image.detach().cpu().squeeze(0))*127.5).clamp(0,255).round().byte().numpy()
    return Image.fromarray(a).convert('RGB')


def evaluate_validation(generated_batches, real_batches, converter, evaluator,
                        output, batch_size=2, dataset_identity=None):
    """Score every cached metric image, at its metric width, against literal labels.

    No training references, sampled target words, stress manifest or per-cell
    budget. Generators are supplied by the caller so OCR consumes the exact
    image stream already used by FID/KID/HWD.
    """
    if not 1 <= batch_size <= 4:
        raise ValueError('OCR batch size must be in [1,4]')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    identity = {'protocol':'full-validation-same-text-v1',
                'dataset':dataset_identity, 'evaluator':evaluator.identity}
    cache_path = output.parent/'validation_real_ocr.json'
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cache.get('identity') != identity:
        cache = {'identity':identity, 'predictions':{}}
    real_cache = cache['predictions']
    rows = []; sentinel = object(); started = time.monotonic()
    def transcripts(batch):
        if hasattr(converter, 'decode_labels'):
            return converter.decode_labels(batch['lbs'], batch['lb_lens'])
        return [''.join(converter.alphabet[int(i)] for i in row[:int(n)] if int(i))
                for row,n in zip(batch['lbs'],batch['lb_lens'])]
    for fake, real in zip_longest(generated_batches, real_batches, fillvalue=sentinel):
        if fake is sentinel or real is sentinel:
            raise ValueError('OCR real/generated validation batch counts differ')
        targets = transcripts(fake)
        if targets != transcripts(real):
            raise ValueError('OCR real/generated validation transcripts differ')
        if len(targets) != len(real['org_imgs']) or len(targets) != len(fake['org_imgs']):
            raise ValueError('OCR validation batch size differs from labels')
        for begin in range(0,len(targets),batch_size):
            ids = range(begin,min(begin+batch_size,len(targets)))
            pictures = [canonical_pil(fake['org_imgs'][i,:,:,:int(fake['org_img_lens'][i])]) for i in ids]
            references = [canonical_pil(real['org_imgs'][i,:,:,:int(real['org_img_lens'][i])]) for i in ids]
            predictions = evaluator.predict(pictures)
            keys = [hashlib.sha256(im.tobytes()+str(im.size).encode()+targets[i].encode()).hexdigest()
                    for im,i in zip(references,ids)]
            for picture,reference,pred,key in zip(pictures,references,predictions,keys):
                if picture.size == reference.size and picture.tobytes() == reference.tobytes():
                    real_cache[key] = pred
            missing = [(j,key) for j,key in enumerate(keys) if key not in real_cache]
            if missing:
                decoded = evaluator.predict([references[j] for j,_ in missing])
                real_cache.update({key:pred for (_,key),pred in zip(missing,decoded)})
            for i,pred,key in zip(ids,predictions,keys):
                rows.append({'index':len(rows),'target':targets[i],'prediction':pred,
                             'real_prediction':real_cache[key],
                             'char_errors':levenshtein(pred,targets[i])})
    if not rows:
        raise ValueError('OCR validation loader is empty')
    if dataset_identity and dataset_identity.get('samples') is not None:
        if len(rows) != int(dataset_identity['samples']):
            raise ValueError('OCR did not cover the complete configured validation dataset')
    result = {**identity, 'samples':len(rows),
              'metrics':recognition_scores([r['target'] for r in rows],[r['prediction'] for r in rows]),
              'real_metrics':recognition_scores([r['target'] for r in rows],[r['real_prediction'] for r in rows]),
              'seconds':time.monotonic()-started}
    cache_path.write_text(json.dumps(cache,indent=2))
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    with (output/'predictions.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return result


class IndependentOCR:
    def __init__(self, device='cpu', model=OCR_MODEL, revision=OCR_REVISION, allow_download=False):
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        self.device = torch.device(device)
        kwargs = {'local_files_only':not allow_download}
        if not Path(model).is_dir(): kwargs['revision'] = revision
        self.processor = TrOCRProcessor.from_pretrained(model, **kwargs)
        self.model = VisionEncoderDecoderModel.from_pretrained(model, **kwargs).float().eval().requires_grad_(False).to(self.device)
        self.identity = {'model':model,'revision':revision,'parameters':sum(p.numel() for p in self.model.parameters()),
                         'metric_version':METRIC_VERSION,'preprocess_version':PREPROCESS_VERSION,
                         'decoding':'greedy, no lexicon, no correction', 'dtype':'float32'}

    def predict(self, images):
        with torch.inference_mode():
            pixels = self.processor(images=images, return_tensors='pt').pixel_values.to(self.device, dtype=torch.float32)
            ids = self.model.generate(pixels, do_sample=False, num_beams=1, max_new_tokens=48)
        return self.processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


def evaluate_transfer(generator, encoder, backbone, converter, train, test, manifest, evaluator,
                      output, batch_size=2, limit=None):
    if not 1 <= batch_size <= 4: raise ValueError('OCR diagnostic batch_size must be in [1,4]')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    cases = manifest['cases'] if limit is None else manifest['cases'][:limit]
    if not cases: raise ValueError('no eligible OCR evaluation cases')
    device = next(generator.parameters()).device
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    start = time.monotonic(); rows = []; sheet = []
    cache_path = output.parent/'ocr_real_cache.json'
    cache_key = hashlib.sha256(json.dumps({'ocr':evaluator.identity,'data':[train.image_fingerprint,test.image_fingerprint],
                    'height':[train.height,test.height],'polarity':[train.source_polarity,test.source_polarity]},sort_keys=True).encode()).hexdigest()
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    real_cache = cache.setdefault(cache_key, {})
    with evaluation_context([generator, encoder, backbone]):
        for begin in range(0,len(cases),batch_size):
            selected = cases[begin:begin+batch_size]
            refs = [(train if row['split']=='train' else test).image(row['index']) for row in selected]
            lengths = torch.tensor([x.size(-1) for x in refs], device=device)
            x = torch.full((len(refs),1,refs[0].size(-2),int(lengths.max())), -1., device=device)
            for i, image in enumerate(refs): x[i,:,:,:image.size(-1)] = image.to(device)
            labels, label_lengths = converter.encode([row['target'] for row in selected])
            z, memory, memory_mask = encoder(x, lengths, backbone, return_memory=True)
            imgs = generator(
                z, labels.to(device), label_lengths.to(device),
                reference_memory=memory, memory_valid_mask=memory_mask,
            )
            pictures = [canonical_pil(imgs[i,:,:,:int(n)*train.height//2]) for i,n in enumerate(label_lengths)]
            predictions = evaluator.predict(pictures)
            missing = [(i,f'{row["split"]}:{row["index"]}') for i,row in enumerate(selected) if f'{row["split"]}:{row["index"]}' not in real_cache]
            if missing:
                real_predictions = evaluator.predict([canonical_pil(refs[i]) for i,_ in missing])
                for (_,key),pred in zip(missing,real_predictions): real_cache[key] = pred
            for i,(row,pred) in enumerate(zip(selected,predictions)):
                rows.append({**row,'prediction':pred,'real_prediction':real_cache[f'{row["split"]}:{row["index"]}'],
                             'char_errors':levenshtein(pred,row['target'])})
                if len(sheet)<12: sheet.append((row,canonical_pil(refs[i]),pictures[i],pred))
    result = {'metrics':recognition_scores([r['target'] for r in rows],[r['prediction'] for r in rows]),
              'real_metrics':recognition_scores([r['reference'] for r in rows],[r['real_prediction'] for r in rows]),
              'groups':{},'samples':len(rows),'evaluator':evaluator.identity,'notes':manifest['notes'],
              'seconds':time.monotonic()-start,'device':str(evaluator.device)}
    for group in sorted({r['scenario'] for r in rows}):
        chosen = [r for r in rows if r['scenario']==group]
        result['groups'][group] = {'samples':len(chosen),**recognition_scores([r['target'] for r in chosen],[r['prediction'] for r in chosen])}
    if device.type=='cuda':
        result['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated(device)/2**20
        result['cuda_peak_reserved_mb'] = torch.cuda.max_memory_reserved(device)/2**20
    cache_path.write_text(json.dumps(cache,indent=2))
    (output/'metrics.json').write_text(json.dumps(result,indent=2))
    with (output/'predictions.csv').open('w',newline='') as handle:
        writer = csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)
    with (output/'metrics.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['cer','wer']);writer.writeheader();writer.writerow(result['metrics'])
    board = Image.new('RGB',(1000,max(1,len(sheet))*100),'white');draw=ImageDraw.Draw(board)
    for i,(row,ref,gen,pred) in enumerate(sheet):
        y=i*100; draw.text((4,y),f'{row["scenario"]} ref={row["reference"]!r} target={row["target"]!r} OCR={pred!r}',fill='black')
        board.paste(ref.resize((min(450,ref.width*2),64)),(4,y+22))
        board.paste(gen.resize((min(500,gen.width*2),64)),(480,y+22))
    board.save(output/'samples.png')
    return result
