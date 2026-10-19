#!/usr/bin/env python3
"""Small independent OCR transfer evaluation; does not run full FID/KID/HWD."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--dataset-root',type=Path,default=ROOT/'data/iam')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/ocr_evaluation')
    parser.add_argument('--manifest',type=Path)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--ocr-device',default=None)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--per-cell',type=int,default=64)
    parser.add_argument('--pairs',type=int,default=32)
    parser.add_argument('--disjoint-writer-splits',action=argparse.BooleanOptionalAction,default=None,
                        help='Declare known disjoint GAN writer splits; numeric IDs may be split-local')
    parser.add_argument('--limit',type=int,default=None)
    parser.add_argument('--lexicon',type=Path,default=ROOT/'data/english_words.txt')
    parser.add_argument('--download-ocr',action='store_true',help='Explicitly allow downloading the pinned independent evaluator')
    parser.add_argument('--prepare-ocr-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.batch_size<=4:parser.error('--batch-size must be 1..4')
    if args.limit is not None and args.limit<1:parser.error('--limit must be positive')
    import torch
    torch.set_num_threads(2)
    from metric.ocr_eval import IndependentOCR, WordCorpus, transfer_manifest, evaluate_transfer
    if args.prepare_ocr_only:
        evaluator=IndependentOCR('cpu',allow_download=args.download_ocr)
        print(json.dumps(evaluator.identity,indent=2));return
    if args.checkpoint is None:parser.error('--checkpoint is required unless --prepare-ocr-only')
    from networks.BigGAN_networks import Generator
    from networks.module import StyleEncoder,StyleBackbone
    from lib.alphabet import strLabelConverter,get_lexicon
    checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    cfg=checkpoint.get('config_snapshot')
    if not cfg:raise ValueError('saved config_snapshot is required to identify the checkpoint architecture')
    height=int(cfg['img_height'])
    generator=Generator(**{**cfg['GenModel'],'init':'none'})
    encoder=StyleEncoder(**{**cfg['EncModel'],'init':'none'})
    backbone=StyleBackbone(**{**cfg['StyBackbone'],'img_height':height,'init':'none'})
    for key,model in [('G',generator),('E',encoder),('B',backbone)]:
        model.load_state_dict(checkpoint.get(key+'_EMA',checkpoint[key]),strict=True)
        model.requires_grad_(False).eval()
    digest=hashlib.sha256()
    with args.checkpoint.open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    identity={'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest.hexdigest(),
              'epoch':checkpoint.get('Epoch'),'height':height,'weights':'EMA' if 'G_EMA' in checkpoint else 'raw'}
    del checkpoint;gc.collect()
    for model in [generator,encoder,backbone]:model.to(args.device)
    def corpus(split):
        names=[f'iam_{split}_{height}.hdf5',f'{"trnval" if split=="train" else "test"}set_words{height}.hdf5']
        path=next((args.dataset_root/name for name in names if (args.dataset_root/name).exists()),None)
        if path is None:raise FileNotFoundError(f'No {split} HDF5 found under {args.dataset_root}: {names}')
        return WordCorpus(path,cfg.get('dataset_source_polarity','white'))
    train,test=corpus('train'),corpus('test')
    converter=strLabelConverter('all')
    lexicon=get_lexicon(str(args.lexicon),converter.alphabet,max_length=cfg['training'].get('max_word_len',20))
    manifest_path=args.manifest or args.output/'manifest.json'
    manifest=transfer_manifest(train,test,lexicon,converter,manifest_path,args.per_cell,args.pairs,
                               cfg['training'].get('max_word_len',20),cfg.get('seed',123456),
                               args.disjoint_writer_splits if args.disjoint_writer_splits is not None else cfg.get('valid',{}).get('ocr_eval_disjoint_writer_splits',False))
    print('OCR cases:',manifest['counts'],'notes:',manifest['notes'],flush=True)
    start=time.monotonic()
    evaluator=IndependentOCR(args.ocr_device or args.device,allow_download=args.download_ocr)
    result=evaluate_transfer(generator,encoder,backbone,converter,train,test,manifest,evaluator,
                             args.output,args.batch_size,args.limit)
    result.update(identity);result['total_seconds']=time.monotonic()-start
    (args.output/'metrics.json').write_text(json.dumps(result,indent=2))
    print('RESULTS',json.dumps({'cer':result['metrics']['cer'],'wer':result['metrics']['wer'],
                               'samples':result['samples'],'seconds':result['total_seconds'],'output':str(args.output)}),flush=True)


if __name__=='__main__':main()
