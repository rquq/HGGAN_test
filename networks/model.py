import torch, os
import wandb
from PIL import Image
from munch import Munch
from itertools import chain
import matplotlib.pyplot as plt
import numpy as np
from distance import levenshtein
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.nn import CTCLoss, CrossEntropyLoss
import torch.distributed as dist
import torch.nn.functional as F
from metric.val_metrics import calculate_fid_kid_is
from metric.mssim_psnr import calculate_mssim_psnr
from networks.utils import _info, set_requires_grad, get_scheduler, idx_to_words, rescale_images, rescale_images2, \
                            words_to_images, ctc_greedy_decoder, extract_all_patches, frozen_bn
from networks.BigGAN_networks import Generator, Discriminator, PatchDiscriminator
from networks.module import Recognizer, WriterIdentifier, StyleEncoder, StyleBackbone
from lib.datasets import get_dataset, get_collect_fn, Hdf5Dataset
from lib.alphabet import strLabelConverter, get_lexicon, get_true_alphabet, Alphabets
from lib.utils import draw_image, get_logger, AverageMeterManager, option_to_string, AverageMeter, plot_heatmap
from networks.rand_dist import prepare_z_dist, prepare_y_dist
from networks.loss import recn_l1_loss, CXLoss, KLloss, contrastive_style_loss
from networks.masking import apply_vertical_stripe_mask, apply_horizontal_stripe_mask, apply_combined_stripe_mask, apply_light_mixed_patch_mask


class EMA(object):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model, beta):
        curr_model_unwrapped = getattr(current_model, 'module', current_model)
        for current_params, ma_params in zip(curr_model_unwrapped.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = old_weight * beta + (1 - beta) * up_weight

    def step_ema(self, ema_model, model, step_start_ema=0):
        if self.step < step_start_ema:
            curr_model_unwrapped = getattr(model, 'module', model)
            ema_model.load_state_dict(curr_model_unwrapped.state_dict())
            return
        beta = min(self.beta, (1 + self.step) / (10 + self.step))
        self.update_model_average(ema_model, model, beta)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


class BaseModel(object):
    def __init__(self, opt, log_root='./'):
        self.opt = opt
        self.local_rank = getattr(opt, 'local_rank', -1)
        self.device = torch.device(opt.device)
        self.models = Munch()
        self.models_ema = Munch()
        self.optimizers = Munch()
        self.log_root = log_root
        self.logger = None
        self.is_resumed_start = False
        alphabet_key = 'rimes_word' if opt.dataset.startswith('rimes') else 'all'
        self.alphabet = Alphabets[alphabet_key]
        self.label_converter = strLabelConverter(alphabet_key)
        self.epoch_start = 1

    @staticmethod
    def unwrap_model(model):
        if model is None:
            return None
        return getattr(model, 'module', model)

    def print(self, info):
        if self.local_rank > 0:
            return
        if self.logger is None:
            print(info)
        else:
            self.logger.info(info)

    def create_logger(self):
        if self.logger:
            return

        if not os.path.exists(self.log_root):
            os.makedirs(self.log_root)

        opt_str = option_to_string(self.opt)
        with open(os.path.join(self.log_root, 'config.txt'), 'w') as f:
            f.writelines(opt_str)
        self.logger = get_logger(self.log_root)

    def info(self, extra=None):
        self.print("RUNDIR: {}".format(self.log_root))
        opt_str = option_to_string(self.opt)
        self.print(opt_str)
        for model in self.models.values():
            self.print(_info(model, ret=True))
        if extra is not None:
            self.print(extra)
        self.print('=' * 20)

    def save(self, tag='best', epoch_done=0, iter_count=None, best_fid=None, **kwargs):
        if self.local_rank > 0:
            return
        ckpt = {}
        for name, model in self.models.items():
            m_unwrapped = self.unwrap_model(model)
            m_dict = m_unwrapped.state_dict()
            ckpt[name] = m_dict
            ckpt[type(m_unwrapped).__name__] = m_dict

        if hasattr(self, 'models_ema') and self.models_ema:
            for name, model_ema in self.models_ema.items():
                m_ema_unwrapped = self.unwrap_model(model_ema)
                m_ema_dict = m_ema_unwrapped.state_dict()
                ckpt[name + '_EMA'] = m_ema_dict
                ckpt[type(m_ema_unwrapped).__name__ + '_EMA'] = m_ema_dict

        if hasattr(self, 'ema_tracker') and self.ema_tracker is not None:
            ckpt['ema_step'] = self.ema_tracker.step

        for key, optim in self.optimizers.items():
            ckpt['OPT.' + key] = optim.state_dict()

        if hasattr(self, 'lr_schedulers') and self.lr_schedulers:
            for key, sched in self.lr_schedulers.items():
                if hasattr(sched, 'state_dict'):
                    ckpt['SCHED.' + key] = sched.state_dict()

        import random
        ckpt['rng_state'] = {
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy': np.random.get_state(),
            'python': random.getstate(),
            'z_dist': self.z.get_state() if hasattr(self, 'z') and hasattr(self.z, 'get_state') else None,
            'y_dist': self.y.get_state() if hasattr(self, 'y') and hasattr(self.y, 'get_state') else None,
            'eval_z_dist': self.eval_z.get_state() if hasattr(self, 'eval_z') and hasattr(self.eval_z, 'get_state') else None,
            'eval_y_dist': self.eval_y.get_state() if hasattr(self, 'eval_y') and hasattr(self.eval_y, 'get_state') else None,
        }

        for key, val in kwargs.items():
            ckpt[key] = val

        ckpt['Epoch'] = epoch_done
        if iter_count is not None:
            ckpt['iter_count'] = iter_count

        # ── Best/Last Checkpoint Saving (Only last_fid_X.pth & best_fid_X.pth) ──
        import shutil, glob
        this_fid = kwargs.get('fid', kwargs.get('FID', getattr(self, 'last_eval_fid', None)))
        if this_fid is None and hasattr(self, 'restored_metadata'):
            this_fid = self.restored_metadata.get('last_eval_fid', self.restored_metadata.get('best_fid', None))
        if this_fid is not None:
            try:
                this_fid = float(this_fid)
                self.last_eval_fid = this_fid
            except Exception:
                this_fid = None

        cached_best = getattr(self, 'best_fid', None)
        if cached_best is None:
            cached_best = getattr(self, 'restored_metadata', {}).get('best_fid', np.inf)
            if cached_best is None: cached_best = np.inf
            try: cached_best = float(cached_best)
            except Exception: cached_best = np.inf

        if best_fid is not None:
            try: best_fid_val = float(best_fid)
            except Exception: best_fid_val = cached_best
        else:
            best_fid_val = cached_best

        is_new_best = (tag == 'best') or (this_fid is not None and this_fid < cached_best)

        if is_new_best and this_fid is not None:
            self.best_fid = this_fid
            best_fid_val = this_fid
            ckpt['best_fid'] = this_fid
        elif best_fid_val < np.inf:
            ckpt['best_fid'] = best_fid_val

        if this_fid is not None:
            ckpt['fid'] = this_fid

        ckpt_dir = os.path.join(self.log_root, self.opt.training.ckpt_dir)
        os.makedirs(ckpt_dir, exist_ok=True)

        # Write once to a temporary file
        tmp_path = os.path.join(ckpt_dir, f".tmp_{tag}.pth")
        torch.save(ckpt, tmp_path)

        if tag == 'last':
            fid_str = f"{this_fid:.4f}" if (this_fid is not None and np.isfinite(this_fid)) else "inf"
            
            for old_last in glob.glob(os.path.join(ckpt_dir, "last_fid_*.pth")) + glob.glob(os.path.join(ckpt_dir, "last.pth")):
                try: os.remove(old_last)
                except Exception: pass

            last_fid_path = os.path.join(ckpt_dir, f"last_fid_{fid_str}.pth")
            shutil.copy(tmp_path, last_fid_path)
            self.print(f"--> Saved last checkpoint: last_fid_{fid_str}.pth")

            if is_new_best:
                best_str = f"{best_fid_val:.4f}" if (best_fid_val is not None and np.isfinite(best_fid_val)) else fid_str
                for old_best in glob.glob(os.path.join(ckpt_dir, "best_fid_*.pth")) + glob.glob(os.path.join(ckpt_dir, "best.pth")):
                    try: os.remove(old_best)
                    except Exception: pass

                best_fid_path = os.path.join(ckpt_dir, f"best_fid_{best_str}.pth")
                shutil.copy(tmp_path, best_fid_path)
                self.print(f"--> Saved new best checkpoint: best_fid_{best_str}.pth (FID: {best_fid_val:.4f})")

        else:
            if is_new_best or tag == 'best':
                best_str = f"{best_fid_val:.4f}" if (best_fid_val is not None and np.isfinite(best_fid_val)) else "inf"
                for old_best in glob.glob(os.path.join(ckpt_dir, "best_fid_*.pth")) + glob.glob(os.path.join(ckpt_dir, "best.pth")):
                    try: os.remove(old_best)
                    except Exception: pass

                best_fid_path = os.path.join(ckpt_dir, f"best_fid_{best_str}.pth")
                shutil.copy(tmp_path, best_fid_path)
                self.print(f"--> Saved best checkpoint: best_fid_{best_str}.pth (FID: {best_fid_val:.4f})")

        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    def restore_rng_state(self, rng=None):
        if rng is None:
            rng = getattr(self, '_restored_rng_state', None)
        if not rng:
            return
        
        if 'torch' in rng and rng['torch'] is not None:
            try:
                torch_rng = rng['torch'].cpu().to(torch.uint8) if isinstance(rng['torch'], torch.Tensor) else rng['torch']
                torch.set_rng_state(torch_rng)
            except Exception as e:
                self.print(f'Restoring torch RNG warning: {e}')

        if 'cuda' in rng and rng['cuda'] is not None and torch.cuda.is_available():
            try:
                cuda_rngs = [r.cpu().to(torch.uint8) if isinstance(r, torch.Tensor) else r for r in rng['cuda']]
                for idx, c_state in enumerate(cuda_rngs):
                    if idx < torch.cuda.device_count():
                        torch.cuda.set_rng_state(c_state, device=idx)
            except Exception as e:
                self.print(f'Restoring CUDA RNG warning: {e}')

        if 'numpy' in rng and rng['numpy'] is not None:
            try:
                import numpy as np
                np.random.set_state(rng['numpy'])
            except Exception as e:
                self.print(f'Restoring numpy RNG warning: {e}')

        if 'python' in rng and rng['python'] is not None:
            try:
                import random
                random.setstate(rng['python'])
            except Exception as e:
                self.print(f'Restoring python RNG warning: {e}')

        if hasattr(self, 'z') and hasattr(self.z, 'set_state') and rng.get('z_dist') is not None:
            try:
                self.z.set_state(rng['z_dist'])
            except Exception as e:
                self.print(f'Restoring z_dist RNG warning: {e}')

        if hasattr(self, 'y') and hasattr(self.y, 'set_state') and rng.get('y_dist') is not None:
            try:
                self.y.set_state(rng['y_dist'])
            except Exception as e:
                self.print(f'Restoring y_dist RNG warning: {e}')

        if hasattr(self, 'eval_z') and hasattr(self.eval_z, 'set_state') and rng.get('eval_z_dist') is not None:
            try:
                self.eval_z.set_state(rng['eval_z_dist'])
            except Exception as e:
                self.print(f'Restoring eval_z RNG warning: {e}')

        if hasattr(self, 'eval_y') and hasattr(self.eval_y, 'set_state') and rng.get('eval_y_dist') is not None:
            try:
                self.eval_y.set_state(rng['eval_y_dist'])
            except Exception as e:
                self.print(f'Restoring eval_y RNG warning: {e}')

    def resolve_resume_path(self, resume_path):
        if not resume_path:
            return None
        if os.path.isfile(resume_path):
            return resume_path
        if isinstance(resume_path, bool) or str(resume_path).lower() in ('true', 'latest'):
            candidate_dir = os.path.join(self.log_root, getattr(self.opt.training, 'ckpt_dir', 'ckpts'))
            if os.path.isdir(candidate_dir):
                import glob
                pths = glob.glob(os.path.join(candidate_dir, "last_fid_*.pth")) + glob.glob(os.path.join(candidate_dir, "last.pth"))
                pths = [p for p in pths if not os.path.basename(p).startswith('.tmp_')]
                if pths:
                    return max(set(pths), key=os.path.getmtime)
        return None

    def load(self, ckpt, map_location=None, modules=None):
        if modules is None:
            modules = []
        elif not isinstance(modules, list):
            modules = [modules]

        resolved_ckpt = self.resolve_resume_path(ckpt)
        if resolved_ckpt:
            ckpt = resolved_ckpt

        if not ckpt or not os.path.exists(ckpt):
            self.print(f'Checkpoint file not found: {ckpt}')
            return 0

        self.print(f'load checkpoint from {ckpt}')
        if map_location is None:
            map_location = 'cpu'
        ckpt_data = torch.load(ckpt, map_location=map_location, weights_only=False)

        if ckpt_data is None:
            return 0

        best_fid = ckpt_data.get('best_fid', ckpt_data.get('fid', None))
        if best_fid is None:
            ckpt_dir_of_file = os.path.dirname(ckpt)
            source_best_pth = os.path.join(ckpt_dir_of_file, 'best.pth')
            if os.path.exists(source_best_pth) and os.path.abspath(source_best_pth) != os.path.abspath(ckpt):
                try:
                    best_data = torch.load(source_best_pth, map_location='cpu', weights_only=False)
                    best_fid = best_data.get('best_fid', best_data.get('fid', None))
                    self.print(f"Restored best_fid={best_fid} from existing best.pth in resume directory")
                except Exception as e:
                    self.print(f"Could not read best_fid from {source_best_pth}: {e}")

        restored_fid = ckpt_data.get('fid', ckpt_data.get('last_eval_fid', None))
        if restored_fid is None and isinstance(ckpt, str):
            import re
            m = re.search(r'_fid_([0-9]+\.[0-9]+)', os.path.basename(ckpt))
            if m:
                try:
                    restored_fid = float(m.group(1))
                except Exception:
                    pass

        if restored_fid is not None:
            try:
                self.last_eval_fid = float(restored_fid)
                self.print(f"Restored last_eval_fid={self.last_eval_fid:.4f} from checkpoint")
            except Exception:
                pass

        self.restored_metadata = {
            'Epoch': ckpt_data.get('Epoch', 0),
            'iter_count': ckpt_data.get('iter_count', None),
            'best_fid': best_fid,
            'last_eval_fid': getattr(self, 'last_eval_fid', None),
            'ema_step': ckpt_data.get('ema_step', None),
        }

        for name, model in self.models.items():
            if len(modules) > 0 and model not in modules:
                continue
            m_unwrapped = self.unwrap_model(model)
            m_name = type(m_unwrapped).__name__
            target_key = name if name in ckpt_data else (m_name if m_name in ckpt_data else None)
            if target_key:
                try:
                    m_unwrapped.load_state_dict(ckpt_data[target_key], strict=False)
                    self.print(f'Loaded weights for {name} using key {target_key}')
                except Exception as e:
                    self.print(f'Load {name} ({target_key}) failed: {e}')
            else:
                self.print(f'Key {name} / {m_name} not found in checkpoint')

        if hasattr(self, 'models_ema') and self.models_ema:
            for name, model_ema in self.models_ema.items():
                ema_key = name + '_EMA'
                alt_ema_key = type(self.unwrap_model(self.models.get(name))).__name__ + '_EMA' if name in self.models else None
                
                target_key = None
                if ema_key in ckpt_data:
                    target_key = ema_key
                elif alt_ema_key and alt_ema_key in ckpt_data:
                    target_key = alt_ema_key

                if target_key:
                    try:
                        m_ema_unwrapped = self.unwrap_model(model_ema)
                        m_ema_unwrapped.load_state_dict(ckpt_data[target_key], strict=False)
                        self.print(f'Loaded EMA weights for {name} using key {target_key}')
                    except Exception as e:
                        self.print(f'Load EMA key {target_key} failed: {e}')
                else:
                    if name in self.models:
                        m_unwrapped = self.unwrap_model(self.models[name])
                        m_ema_unwrapped = self.unwrap_model(model_ema)
                        m_ema_unwrapped.load_state_dict(m_unwrapped.state_dict())
                        self.print(f'Initialized EMA weights for {name} from active model')

        for key in self.optimizers.keys():
            opt_key = 'OPT.' + key
            if opt_key in ckpt_data:
                try:
                    self.optimizers[key].load_state_dict(ckpt_data[opt_key])
                    for state in self.optimizers[key].state.values():
                        for k_s, v_s in state.items():
                            if isinstance(v_s, torch.Tensor):
                                state[k_s] = v_s.to(self.device)
                    self.print(f'Loaded optimizer state for OPT.{key}')
                except Exception as e:
                    # Fallback for legacy checkpoints: verify shape matching for each parameter
                    try:
                        import copy
                        adapted_opt_dict = copy.deepcopy(ckpt_data[opt_key])
                        curr_params = self.optimizers[key].param_groups[0]['params']
                        saved_param_ids = adapted_opt_dict['param_groups'][0]['params']
                        adapted_opt_dict['param_groups'][0]['params'] = list(range(len(curr_params)))
                        new_state = {}
                        for idx, p_curr in enumerate(curr_params):
                            curr_shape = tuple(p_curr.shape)
                            if idx < len(saved_param_ids):
                                p_id = saved_param_ids[idx]
                                if p_id in ckpt_data[opt_key].get('state', {}):
                                    item = ckpt_data[opt_key]['state'][p_id]
                                    if 'exp_avg' in item and tuple(item['exp_avg'].shape) == curr_shape:
                                        new_state[idx] = item
                        adapted_opt_dict['state'] = new_state
                        self.optimizers[key].load_state_dict(adapted_opt_dict)
                        for state in self.optimizers[key].state.values():
                            for k_s, v_s in state.items():
                                if isinstance(v_s, torch.Tensor):
                                    state[k_s] = v_s.to(self.device)
                        self.print(f'Loaded adapted optimizer state for OPT.{key} (legacy checkpoint compatibility)')
                    except Exception as inner_e:
                        self.print(f'Load OPT.{key} failed: {inner_e}')

        if hasattr(self, 'lr_schedulers') and self.lr_schedulers:
            for key in self.lr_schedulers.keys():
                sched_key = 'SCHED.' + key
                if sched_key in ckpt_data:
                    try:
                        self.lr_schedulers[key].load_state_dict(ckpt_data[sched_key])
                        if hasattr(self.lr_schedulers[key], 'get_last_lr') and key in self.optimizers:
                            lrs = self.lr_schedulers[key].get_last_lr()
                            for param_group, lr in zip(self.optimizers[key].param_groups, lrs):
                                param_group['lr'] = lr
                        self.print(f'Loaded scheduler state for SCHED.{key}')
                    except Exception as e:
                        self.print(f'Load SCHED.{key} failed: {e}')
        self._ckpt_sched_data = {k: v for k, v in ckpt_data.items() if k.startswith('SCHED.')}

        if 'rng_state' in ckpt_data:
            self._restored_rng_state = ckpt_data['rng_state']
            self.restore_rng_state(self._restored_rng_state)


        if hasattr(self, 'ema_tracker') and self.ema_tracker is not None and self.restored_metadata.get('ema_step') is not None:
            self.ema_tracker.step = self.restored_metadata['ema_step']
            self.print(f'Loaded EMA tracker step={self.ema_tracker.step}')

        epoch = self.restored_metadata['Epoch']
        self.is_resumed_start = True
        del ckpt_data
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return epoch

    def set_mode(self, mode='eval'):
        for model in self.models.values():
            if mode == 'eval':
                model.eval()
            elif mode == 'train':
                model.train()
            else:
                raise NotImplementedError()
        if hasattr(self, 'models_ema') and self.models_ema:
            for model_ema in self.models_ema.values():
                if mode == 'eval':
                    model_ema.eval()
                elif mode == 'train':
                    model_ema.train()

    def validate(self, *args, **kwargs):
        raise NotImplementedError()


    def train(self):
        raise NotImplementedError()


class AdversarialModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(AdversarialModel, self).__init__(opt, log_root)

        self.lexicon = get_lexicon(self.opt.training.lexicon,
                                   get_true_alphabet(opt.dataset),
                                   max_length=self.opt.training.max_word_len)
        self.max_valid_image_width = self.opt.char_width * self.opt.training.max_word_len
        self.vae_mode = self.opt.training.vae_mode
        self.collect_fn = get_collect_fn(self.opt.training.sort_input, sort_style=True)
        self.inception_model = None
        self.valid_real_stats = None
        dataset = get_dataset(opt.dataset, opt.training.dset_split,
                              recogn_aug=True, wid_aug=True, process_style=True)
        if self.local_rank > -1:
            from torch.utils.data.distributed import DistributedSampler
            self.train_sampler = DistributedSampler(dataset, num_replicas=None, rank=self.local_rank, shuffle=True)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        self.train_loader = DataLoader(
            dataset,
            batch_size=opt.training.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            collate_fn=self.collect_fn,
            num_workers=4,
            drop_last=True,
            pin_memory=(self.device.type == 'cuda'),
            persistent_workers=True,
            worker_init_fn=seed_worker
        )

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split,
                        recogn_aug=False, wid_aug=False, process_style=True),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
            pin_memory=(self.device.type == 'cuda')
        )

        self.tst_loader2 = DataLoader(
            get_dataset(opt.dataset, opt.training.dset_split,
                        recogn_aug=False, wid_aug=False, process_style=True),
            batch_size=opt.training.eval_batch_size // 2,
            shuffle=True,
            collate_fn=self.collect_fn,
            pin_memory=(self.device.type == 'cuda')
        )

        self.models = None

    def train(self):
        raise NotImplementedError()

    def sample_images(self, iteration_done=0):
        self.set_mode('eval')

        device = self.device
        batchA = next(iter(self.tst_loader))
        batchB = next(iter(self.tst_loader2))
        batch = Hdf5Dataset.merge_batch(batchA, batchB, device)

        real_imgs, real_img_lens = batch['style_imgs'], batch['style_img_lens']
        real_lbs, real_lb_lens = batch['lbs'], batch['lb_lens']

        with torch.no_grad():
            self.eval_z.sample_()
            eval_z_in = self.eval_z

            recn_imgs = None
            if 'E' in self.models:
                enc_z = self.models.E(real_imgs, real_img_lens, self.models.B)
                recn_imgs = self.models.G(enc_z, real_lbs, real_lb_lens)

            fake_real_imgs = self.models.G(eval_z_in, real_lbs, real_lb_lens)

            self.eval_y.sample_()
            sampled_words = idx_to_words(self.eval_y, self.lexicon, 0,
                                         self.opt.training.capitalize_ratio,
                                         self.opt.training.blank_ratio)
            sampled_words[-2] = sampled_words[-1]
            fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words)
            fake_lbs, fake_lb_lens = fake_lbs.to(device), fake_lb_lens.to(device)
            fake_imgs = self.models.G(eval_z_in, fake_lbs, fake_lb_lens)
            style_imgs = self.models.G(enc_z, fake_lbs, fake_lb_lens)

            tensors_to_pad = [real_imgs, fake_real_imgs, fake_imgs, style_imgs]
            if recn_imgs is not None:
                tensors_to_pad.append(recn_imgs)
            max_img_len = max([t.size(-1) for t in tensors_to_pad])
            img_shape = [real_imgs.size(2), max_img_len, real_imgs.size(1)]

            real_imgs = F.pad(real_imgs, [0, max_img_len - real_imgs.size(-1), 0, 0], value=-1.)
            fake_real_imgs = F.pad(fake_real_imgs, [0, max_img_len - fake_real_imgs.size(-1), 0, 0], value=-1.)
            fake_imgs = F.pad(fake_imgs, [0, max_img_len - fake_imgs.size(-1), 0, 0], value=-1.)
            recn_imgs = F.pad(recn_imgs, [0, max_img_len - recn_imgs.size(-1), 0, 0], value=-1.) \
                        if recn_imgs is not None else None
            style_imgs = F.pad(style_imgs, [0, max_img_len - style_imgs.size(-1), 0, 0], value=-1.)

            real_words = self.label_converter.decode(real_lbs, real_lb_lens)
            real_labels = words_to_images(real_words, *img_shape)
            rand_labels = words_to_images(sampled_words, *img_shape)

            try:
                sample_img_list = [real_labels.cpu(), real_imgs.cpu(), fake_real_imgs.cpu(),
                                   fake_imgs.cpu(), style_imgs.cpu(), rand_labels.cpu()]
                if recn_imgs is not None:
                    sample_img_list.insert(2, recn_imgs.cpu())
                sample_imgs = torch.cat(sample_img_list, dim=2).repeat(1, 3, 1, 1)
                res_img = draw_image(1 - sample_imgs.data, nrow=self.opt.training.sample_nrow, normalize=True)
                save_path = os.path.join(self.log_root, self.opt.training.sample_dir,
                                         'iter_{}.png'.format(iteration_done))
                im = Image.fromarray(res_img)
                im.save(save_path)
                self.print(f"--> Saved sample image: iter_{iteration_done}.png")

                import wandb as _wandb
                if _wandb.run:
                    _wandb.log({'samples/generated': _wandb.Image(res_img, caption=f'iter {iteration_done}')},
                               step=iteration_done)
            except RuntimeError as e:
                self.print(e)

    def image_generator(self, style_dloader, use_rand_corpus=False, style_guided=True, n_repeats=1):
        device = self.device
        word_idx_sampler = None
        if use_rand_corpus:
            word_idx_sampler = prepare_y_dist(style_dloader.batch_size,
                                              len(self.lexicon),
                                              self.device,
                                              seed=self.opt.seed)

        if style_guided and not use_rand_corpus:
            n_repeats = 1

        with torch.no_grad():
            for _ in range(n_repeats):
                for batch in style_dloader:
                    fake_batch = {}
                    style_imgs, style_img_lens = batch['style_imgs'].to(device), batch['style_img_lens'].to(device)
                    style_lbs, style_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)
                    if use_rand_corpus:
                        word_idx_sampler.sample_()
                        sampled_words = idx_to_words(word_idx_sampler[:style_imgs.size(0)],
                                                     self.lexicon, 0, self.opt.training.capitalize_ratio,
                                                     blank_ratio=0)
                        content_lbs, content_lb_lens = self.label_converter.encode(sampled_words)
                    else:
                        content_lbs, content_lb_lens = style_lbs, style_lb_lens

                    fake_batch['lbs'], fake_batch['lb_lens'] = content_lbs.to(device), content_lb_lens.to(device)

                    if style_guided:
                        enc_z = self.models.E(style_imgs.to(device), style_img_lens.to(device), self.models.B)
                    else:
                        enc_z = torch.randn(style_lb_lens.size(0), self.models.G.style_dim, self.models.G.style_dim).to(device)

                    fake_batch['style_imgs'] = self.models.G(enc_z, content_lbs, content_lb_lens)
                    fake_batch['style_img_lens'] = fake_batch['lb_lens'] * self.opt.char_width
                    fake_batch['wids'] = batch['wids']

                    fake_batch['org_imgs'], fake_batch['org_img_lens'] =\
                                        rescale_images(fake_batch['style_imgs'],
                                        fake_batch['style_img_lens'],
                                        batch['org_img_lens'])

                    yield fake_batch

    def validate(self, style_guided=True, test_stage=False, *args, **kwargs):
        use_ema = getattr(self, 'use_ema', False)
        if use_ema:
            active_G = self.models.G
            active_E = self.models.E
            active_B = self.models.B
            self.models.G = self.models_ema.G
            self.models.E = self.models_ema.E
            self.models.B = self.models_ema.B

        self.set_mode('eval')

        try:
            # OPTIMIZATION: Cache validation DataLoader to avoid worker startup/shutdown overhead
            if not hasattr(self, 'eval_dloader') or self.eval_dloader is None:
                self.eval_dloader = DataLoader(
                    get_dataset(self.opt.valid.dset_name, self.opt.valid.dset_split, process_style=True),
                    collate_fn=self.collect_fn,
                    batch_size=self.opt.valid.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=(self.device.type == 'cuda'),
                    persistent_workers=True,
                    worker_init_fn=seed_worker
                )
            eval_dloader = self.eval_dloader

            if 'E' not in self.models:
                style_guided = False
                n_rand_repeat = 1
            else:
                n_rand_repeat = 1 if style_guided and not self.opt.valid.use_rand_corpus \
                                  else self.opt.valid.n_rand_repeat

            def get_generator():
                generator = self.image_generator(eval_dloader, self.opt.valid.use_rand_corpus,
                                                 style_guided, n_rand_repeat)
                return generator

            # OPTIMIZATION: Pre-generate and cache fake image batches on CPU.
            # We compress images to int8 and drop style_imgs if not test_stage to fit within tight 15GB RAM limits.
            def batch_to_cpu(batch):
                cpu_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        if k == 'style_imgs' and not test_stage:
                            continue
                        if k in ['org_imgs', 'style_imgs']:
                            cpu_batch[k] = (v.cpu().clamp(-1.0, 1.0) * 127.0).round().to(torch.int8)
                        else:
                            cpu_batch[k] = v.cpu()
                    else:
                        cpu_batch[k] = v
                return cpu_batch

            self.print("Generating and caching validation fake images...")
            generator_list = [batch_to_cpu(b) for b in get_generator()]

            cached_decompressed_list = None
            def get_cached_generator():
                nonlocal cached_decompressed_list
                if cached_decompressed_list is None:
                    cached_decompressed_list = []
                    for batch in generator_list:
                        decompressed = {}
                        for k, v in batch.items():
                            if k in ['org_imgs', 'style_imgs'] and isinstance(v, torch.Tensor):
                                decompressed[k] = (v.to(torch.float32) / 127.0).pin_memory()
                            else:
                                decompressed[k] = v
                        cached_decompressed_list.append(decompressed)
                return cached_decompressed_list

            if not hasattr(self, 'valid_real_stats') or self.valid_real_stats is None:
                from metric.val_metrics import calculate_activation_statistics, InceptionV3
                self.print("Precalculating validation set statistics...")
                block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
                if self.inception_model is None:
                    self.inception_model = InceptionV3([block_idx]).to(self.device).eval()
                self.valid_real_stats = calculate_activation_statistics(eval_dloader, len(eval_dloader), 
                                                                       self.inception_model, self.opt.valid.dims, 
                                                                       self.device, crop=not test_stage)
                                                                       
            from metric.val_metrics import calculate_hwd_score

            if test_stage:
                res = calculate_fid_kid_is(self.opt.valid, eval_dloader, get_cached_generator(), n_rand_repeat, 
                                         self.device, real_stats=self.valid_real_stats, inceptionV3_model=self.inception_model)
            else:
                res = calculate_fid_kid_is(self.opt.valid, eval_dloader, get_cached_generator(), n_rand_repeat, 
                                         self.device, crop=True, real_stats=self.valid_real_stats, inceptionV3_model=self.inception_model)

            if test_stage:
                if not self.opt.valid.use_rand_corpus:
                    psnr_mssim = calculate_mssim_psnr(eval_dloader, get_cached_generator())
                    res['psnr'] = psnr_mssim['psnr']
                    res['mssim'] = psnr_mssim['mssim']
                if style_guided:
                    wier = self.validate_wid(get_cached_generator(), real_dloader=eval_dloader, split=self.opt.valid.dset_split)
                    res['wier'] = wier

            if getattr(self.opt.valid, 'validate_ocr', True):
                res['cer'], res['wer'] = self.validate_ocr(get_cached_generator(), n_iters=len(eval_dloader) * n_rand_repeat)

            if getattr(self.opt.valid, 'validate_hwd', True):
                # OPTIMIZATION: Cache real HWD features to avoid reprocessing real images every epoch.
                if not hasattr(self, 'valid_real_hwd_features') or self.valid_real_hwd_features is None:
                    if not hasattr(self, 'valid_real_hwd_dataset') or self.valid_real_hwd_dataset is None:
                        self.print("Precalculating real image features for HWD...")
                        from metric.val_metrics import ImageListDataset, batch_tensor_to_pil_list
                        real_imgs_list = []
                        real_authors_list = []
                        for idx, batch in enumerate(tqdm(eval_dloader, desc='HWD Real Images')):
                            imgs = batch['org_imgs']
                            lens = batch['org_img_lens']
                            wids = batch.get('wids', torch.arange(imgs.size(0)))
                            pil_imgs = batch_tensor_to_pil_list(imgs, lens)
                            real_imgs_list.extend(pil_imgs)
                            for i in range(imgs.size(0)):
                                real_authors_list.append(str(wids[i].item()))
                        self.valid_real_hwd_dataset = ImageListDataset(real_imgs_list, real_authors_list)
                    
                    from metric.val_metrics import HWDScore
                    hwd_scorer = HWDScore(batchsize=64).to(self.device)
                    self.valid_real_hwd_features = hwd_scorer.digest(self.valid_real_hwd_dataset)
                    self.valid_real_hwd_dataset = None
                    import gc
                    gc.collect()

                hwd_val = calculate_hwd_score(eval_dloader, get_cached_generator(), n_rand_repeat, self.device, real_features=self.valid_real_hwd_features)
                res['hwd'] = hwd_val

            if getattr(self.opt.valid, 'validate_cmmd', True):
                current_epoch = kwargs.get('current_epoch', None)
                every_n = getattr(self.opt.valid, 'validate_cmmd_every_n_epochs', 3)
                should_run_cmmd = test_stage or (current_epoch is None)
                if not should_run_cmmd:
                    should_run_cmmd = (current_epoch % every_n == 0)
                    
                if should_run_cmmd:
                    from metric.val_metrics import calculate_cmmd_score, compute_real_embeddings
                    if not hasattr(self, 'cmmd_embedding_model') or self.cmmd_embedding_model is None:
                        from metric.val_metrics import ClipEmbeddingModel
                        self.cmmd_embedding_model = ClipEmbeddingModel(self.device)
                    if not hasattr(self, 'real_cmmd_embeddings') or self.real_cmmd_embeddings is None:
                        import os
                        import numpy as np
                        cache_dir = "./pretrained"
                        safe_dset_split = self.opt.valid.dset_split.replace('/', '_').replace('\\', '_').replace('.', '_')
                        cache_path = os.path.join(cache_dir, f"real_cmmd_{self.opt.valid.dset_name}_{safe_dset_split}.npy")
                        if os.path.exists(cache_path):
                            self.print(f"Loading cached real CMMD embeddings from {cache_path}...")
                            self.real_cmmd_embeddings = np.load(cache_path)
                        else:
                            self.print("Precalculating real image embeddings for CMMD...")
                            self.real_cmmd_embeddings = compute_real_embeddings(
                                eval_dloader, self.cmmd_embedding_model, device=self.device
                            )
                            try:
                                os.makedirs(cache_dir, exist_ok=True)
                                np.save(cache_path, self.real_cmmd_embeddings)
                                self.print(f"Saved real CMMD embeddings to cache: {cache_path}")
                            except Exception as e:
                                self.print(f"Could not save real CMMD embeddings cache: {e}")
                    cmmd_val = calculate_cmmd_score(
                        eval_dloader, 
                        get_cached_generator(), 
                        n_rand_repeat, 
                        self.device,
                        real_embeddings=self.real_cmmd_embeddings,
                        embedding_model=self.cmmd_embedding_model
                    )
                    res['cmmd'] = cmmd_val

            import gc
            gc.collect()
            torch.cuda.empty_cache()
        finally:
            if use_ema:
                self.models.G = active_G
                self.models.E = active_E
                self.models.B = active_B

        return res

    def validate_ocr(self, dloader, n_iters):
        self.set_mode('eval')
        # Use the already loaded recognizer from self.models instead of creating a new one
        # to avoid redundant memory allocation and potential OOM.
        recognizer = self.unwrap_model(self.models.R)
        
        ctc_len_scale = recognizer.len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0

        with torch.no_grad():
            for i, batch in tqdm(enumerate(dloader), total=n_iters):
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device, non_blocking=True), batch['style_img_lens'].to(self.device, non_blocking=True)
                logits = recognizer(real_imgs, real_img_lens)
                logits = torch.nn.functional.softmax(logits, dim=2).detach()

                logits = logits.cpu().numpy()
                word_preds = []
                for logit, img_len in zip(logits, batch['style_img_lens'].cpu().numpy()):
                    label = ctc_greedy_decoder(logit[:img_len // ctc_len_scale])
                    word_preds.append(self.label_converter.decode(label))
                word_reals = self.label_converter.decode(batch['lbs'], batch['lb_lens'])
                for word_pred, word_real in zip(word_preds, word_reals):
                    char_tran = levenshtein(word_pred, word_real)
                    char_trans += char_tran
                    total_chars += len(word_real)
                    total_words += 1
                    if char_tran > 0:
                        word_trans += 1

        cer = char_trans * 1.0 / max(total_chars, 1)
        wer = word_trans * 1.0 / max(total_words, 1)
        self.print('CER:{:.4f}  WER:{:.4f}'.format(cer, wer))
        return cer, wer

    def validate_wid(self, generator, real_dloader, split='test'):
        if split == 'test':
            assert os.path.exists(self.opt.valid.pretrained_test_w)
            w_dict = torch.load(self.opt.valid.pretrained_test_w, map_location=self.device, weights_only=False)
            test_writer = WriterIdentifier(**self.opt.valid.test_wid_model).to(self.device)
            test_writer.load_state_dict(w_dict['WriterIdentifier'], strict=False)
            test_writer_backbone = StyleBackbone(**self.opt.StyBackbone).to(self.device)
            test_writer_backbone.load_state_dict(w_dict['StyleBackbone'], strict=False)
            self.print(f'load pretrained test_writer_identifier: {self.opt.valid.pretrained_test_w}')
            writer_identifier = test_writer
            writer_backbone = test_writer_backbone
        else:
            # OPTIMIZATION: Use the already loaded WriterIdentifier and StyleBackbone
            # from self.models instead of creating a new copy to avoid redundant VRAM allocation and OOM.
            writer_identifier = self.unwrap_model(self.models.W)
            writer_backbone = self.unwrap_model(self.models.B)
            self.print('Using already loaded writer identifier and style backbone')

        writer_identifier.eval(), writer_backbone.eval()
        with torch.no_grad():
            n_iters = len(real_dloader)

            acc_counts = 0.
            total_counts = 0.
            for i, (batch_real, batch_fake) \
                in tqdm(enumerate(zip(real_dloader, generator)), total=n_iters):
                # predicting pesudo labels
                real_wid_logits = writer_identifier(batch_real['style_imgs'].to(self.device, non_blocking=True),
                                                batch_real['style_img_lens'].to(self.device, non_blocking=True),
                                                writer_backbone)
                _, real_preds = torch.max(real_wid_logits.data, dim=1)

                # predicting pesudo labels
                fake_wid_logits = writer_identifier(batch_fake['style_imgs'].to(self.device, non_blocking=True),
                                                batch_fake['style_img_lens'].to(self.device, non_blocking=True),
                                                writer_backbone)
                _, fake_preds = torch.max(fake_wid_logits.data, dim=1)
                acc_counts += real_preds.eq(fake_preds.to(self.device)).sum().item()
                total_counts += real_preds.size(0)

            wier = 1.0 - acc_counts / total_counts if total_counts > 0 else 1.0

        self.print('WID_wier:{:.2f}'.format(wier))
        return wier

    def eval_interp(self):
        self.set_mode('eval')

        with torch.no_grad():
            interp_num = self.opt.test.interp_num
            nrow, ncol = 1, interp_num
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs).unsqueeze(0)
                fake_lb_lens = torch.IntTensor([len(text)])

                num_tokens = getattr(self.opt.EncModel, 'num_style_tokens', 32)
                style_dim = getattr(self.opt.EncModel, 'style_dim', 32)
                style0 = torch.randn((1, num_tokens, style_dim))
                style1 = torch.randn(style0.size())

                styles = [torch.lerp(style0, style1, i / (interp_num - 1)) for i in range(interp_num)]
                styles = torch.cat(styles, dim=0).float().to(self.device)

                fake_lbs, fake_lb_lens = fake_lbs.repeat(nrow * ncol, 1).to(self.device),\
                                         fake_lb_lens.repeat(nrow * ncol).to(self.device)
                gen_imgs = self.models.G(styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                plt.figure()
                for i in range(nrow * ncol):
                    plt.subplot(nrow, ncol, i + 1)
                    plt.imshow(gen_imgs[i], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_style(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset(self.opt.dataset, self.opt.training.dset_split, process_style=True),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn,
            drop_last=False
        )

        with torch.no_grad():
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                batch = next(iter(tst_loader))
                imgs, img_lens, lbs, lb_lens = \
                    batch['style_imgs'], batch['style_img_lens'], batch['lbs'], batch['lb_lens']
                real_imgs, real_img_lens = imgs.to(self.device), img_lens.to(self.device)
                fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                nrow = batch['style_imgs'].size(0)
                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.B)
                S, D = enc_styles.size(1), enc_styles.size(2)
                enc_styles = enc_styles.unsqueeze(1).repeat(1, ncol, 1, 1).view(nrow * ncol, S, D)

                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                gen_imgs, gen_img_lens = rescale_images2(gen_imgs, fake_lb_lens * self.opt.char_width, fake_lb_lens,
                                           batch['org_img_lens'].repeat_interleave(ncol).to(self.device),
                                           batch['lb_lens'].repeat_interleave(ncol).to(self.device))
                gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                max_w = max(gen_imgs.shape[-1], batch['org_imgs'].size(-1))
                pad_real_w = max_w - batch['org_imgs'].size(-1)
                real_imgs = torch.nn.functional.pad(batch['org_imgs'],
                                                    [0, pad_real_w, 0, 0],
                                                    mode='constant', value=-1)
                real_imgs = (1 - real_imgs).squeeze(1).cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 1)
                    # plt.imshow(real_imgs[i, :, :real_img_lens[i]], cmap='gray')
                    plt.imshow(real_imgs[i], cmap='gray')
                    plt.axis('off')
                    for j in range(ncol):
                        plt.subplot(nrow, 1 + ncol, i * (1 + ncol) + 2 + j)
                        # plt.imshow(gen_imgs[i * ncol + j, :, :gen_img_lens[i * ncol + j]], cmap='gray')
                        plt.imshow(gen_imgs[i * ncol + j], cmap='gray')
                        plt.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_rand(self):
        self.set_mode('eval')

        with torch.no_grad():
            nrow, ncol = self.opt.test.nrow, 2
            rand_z = prepare_z_dist(nrow, self.opt.EncModel.style_dim, self.device, num_tokens=self.opt.EncModel.style_dim)
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                texts = text.split(' ')
                ncol = len(texts)
                fake_lbs, fake_lb_lens = self.label_converter.encode(texts)

                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow, ).to(self.device)

                rand_z.sample_()
                rand_styles = rand_z.unsqueeze(1).repeat(1, ncol, 1, 1).view(nrow * ncol, rand_z.size(1), -1)
                gen_imgs = self.models.G(rand_styles, fake_lbs, fake_lb_lens)
                gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                plt.figure()
                for i in range(nrow):
                    for j in range(ncol):
                        ax = plt.subplot(nrow, ncol, i * ncol + 1 + j)
                        gen_img = gen_imgs[i * ncol + j]
                        ax.imshow(gen_img, cmap='gray')
                        ax.axis('off')
                plt.tight_layout()
                plt.show()

    def eval_text(self):
        self.set_mode('eval')

        tst_loader = DataLoader(
            get_dataset(self.opt.dataset, self.opt.training.dset_split, process_style=True),
            batch_size=self.opt.test.nrow,
            shuffle=True,
            collate_fn=self.collect_fn,
            drop_last=False
        )

        def get_space_index(text):
            idxs = []
            for i, ch in enumerate(text):
                if ch == ' ':
                    idxs.append(i)
            return idxs

        with torch.no_grad():
            while True:
                text = input('input text: ')
                if len(text) == 0:
                    break

                batch = next(iter(tst_loader))
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device), batch['style_img_lens'].to(self.device)
                fake_lbs = self.label_converter.encode(text)
                fake_lbs = torch.LongTensor(fake_lbs)
                fake_lb_lens = torch.IntTensor([len(text)])

                nrow = real_imgs.size(0)
                fake_lbs = fake_lbs.repeat(nrow, 1).to(self.device)
                fake_lb_lens = fake_lb_lens.repeat(nrow,).to(self.device)
                enc_styles = self.models.E(real_imgs, real_img_lens, self.models.B)

                real_imgs = (1 - real_imgs).squeeze(1).cpu().numpy() * 127
                gen_imgs = self.models.G(enc_styles, fake_lbs, fake_lb_lens)
                space_indexs = get_space_index(text)
                for idx in space_indexs:
                    gen_imgs[:, :, idx * self.opt.char_width: (idx + 1) * self.opt.char_width] = -1
                gen_imgs, gen_img_lens = rescale_images2(gen_imgs, fake_lb_lens * self.opt.char_width, fake_lb_lens,
                                           batch['org_img_lens'].to(self.device),
                                           batch['lb_lens'].to(self.device))
                gen_imgs = (1 - gen_imgs).squeeze(1).cpu().numpy() * 127
                plt.figure()

                for i in range(nrow):
                    plt.subplot(nrow * 2, 1, i * 2 + 1)
                    plt.imshow(real_imgs[i, :, :real_img_lens[i]], cmap='gray')
                    plt.axis('off')
                    plt.subplot(nrow * 2, 1, i * 2 + 2)
                    plt.imshow(gen_imgs[i, :, :gen_img_lens[i]], cmap='gray')
                    plt.axis('off')
                plt.tight_layout()
                plt.show()


class GlobalLocalAdversarialModel(AdversarialModel):
    def __init__(self, opt, log_root='./'):
        super(GlobalLocalAdversarialModel, self).__init__(opt, log_root)

        device = self.device

        generator = Generator(**opt.GenModel).to(device)
        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        style_encoder = StyleEncoder(**opt.EncModel).to(device)
        writer_identifier = WriterIdentifier(**opt.WidModel).to(device)
        discriminator = Discriminator(**opt.DiscModel).to(device)
        patch_discriminator = PatchDiscriminator(**opt.PatchDiscModel).to(device)
        recognizer = Recognizer(**opt.OcrModel).to(device)

        self.models = Munch(
            G=generator,
            D=discriminator,
            P=patch_discriminator,
            R=recognizer,
            E=style_encoder,
            W=writer_identifier,
            B=style_backbone,
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')
        self.classify_loss = CrossEntropyLoss()
        self.contextual_loss = CXLoss()
        from networks.loss import GramStyleLoss
        self.gram_loss = GramStyleLoss()

    def train(self):
        self.info()

        # ── WandB init (master process only) ──────────────────────────────
        _is_master = self.local_rank < 1
        if _is_master and not getattr(self.opt, 'no_wandb', False):
            try:
                import wandb as _wandb
                # Get branchname and dates dynamically
                import subprocess
                from datetime import datetime
                branchname = None
                try:
                    branchname = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
                except Exception:
                    pass
                if not branchname or branchname == 'HEAD':
                    try:
                        curr_dir = os.path.abspath(os.getcwd())
                        for _ in range(5):
                            head_path = os.path.join(curr_dir, '.git', 'HEAD')
                            if os.path.exists(head_path):
                                with open(head_path, 'r') as f:
                                    content = f.read().strip()
                                if content.startswith('ref:'):
                                    branchname = content.split('/')[-1]
                                else:
                                    branchname = content[:7]
                                break
                            curr_dir = os.path.dirname(curr_dir)
                    except Exception:
                        pass
                folder_branch = None
                try:
                    parts = os.path.abspath(__file__).split(os.sep)
                    if len(parts) >= 3:
                        folder_branch = parts[-3]
                except Exception:
                    pass
                if branchname in [None, 'main', 'master', 'HEAD']:
                    if folder_branch in ['main', 'dev', 'random_crop_recog', 'classic_optimized', 'HiGANplus', 'higanplus']:
                        branchname = folder_branch
                if not branchname:
                    branchname = 'random_crop_recog'
                
                run_name = f"{branchname}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                wandb_key = os.environ.get('WANDB_API_KEY')
                if not wandb_key:
                    for path_candidate in [
                        '/home/quq/machineLearning/HTG/wandb_key.txt',
                        '/kaggle/working/wandb_key.txt',
                        '../../wandb_key.txt',
                        '../wandb_key.txt',
                        './wandb_key.txt'
                    ]:
                        if os.path.exists(path_candidate):
                            try:
                                with open(path_candidate, 'r') as f:
                                    wandb_key = f.read().strip()
                                if wandb_key:
                                    break
                            except Exception:
                                pass
                if wandb_key:
                    _wandb.login(key=wandb_key)
                else:
                    _wandb.login()
                _wandb.init(
                    project='HiGANplus',
                    name=run_name,
                    config=vars(self.opt) if hasattr(self.opt, '__dict__') else dict(self.opt),
                    resume='allow',
                )
            except Exception as e:
                self.print(f"WandB initialization skipped or failed: {e}")

        opt = self.opt
        self.z = prepare_z_dist(opt.training.batch_size, opt.EncModel.style_dim, self.device,
                                seed=self.opt.seed, num_tokens=opt.EncModel.style_dim)
        self.y = prepare_y_dist(opt.training.batch_size, len(self.lexicon), self.device, seed=self.opt.seed)

        self.eval_z = prepare_z_dist(opt.training.eval_batch_size, opt.EncModel.style_dim, self.device,
                                     seed=self.opt.seed, num_tokens=opt.EncModel.style_dim)
        self.eval_y = prepare_y_dist(opt.training.eval_batch_size, len(self.lexicon), self.device,
                                     seed=self.opt.seed)

        self.optimizers = Munch(
            G=torch.optim.Adam(chain(self.models.G.parameters(), self.models.E.parameters()),
                               lr=opt.training.lr, betas=(opt.training.adam_b1, opt.training.adam_b2)),
            D=torch.optim.Adam(chain(self.models.D.parameters(), self.models.P.parameters()),
                               lr=opt.training.lr,
                               betas=(opt.training.adam_b1, opt.training.adam_b2)),
        )

        # Initialize EMA models
        self.use_ema = getattr(opt.training, 'update_ema', False)
        if self.use_ema:
            import copy
            self.ema_beta = getattr(opt.training, 'ema_beta', 0.999)
            self.print(f"EMA is enabled with beta={self.ema_beta}. Initializing EMA models...")
            self.models_ema.G = copy.deepcopy(self.models.G)
            self.models_ema.E = copy.deepcopy(self.models.E)
            self.models_ema.B = copy.deepcopy(self.models.B)
            self.models_ema.G.requires_grad_(False)
            self.models_ema.E.requires_grad_(False)
            self.models_ema.B.requires_grad_(False)
            self.ema_tracker = EMA(self.ema_beta)

        epoch_done = 1
        resume_path = getattr(self.opt.training, 'resume', None)
        if not resume_path or not os.path.exists(resume_path):
            resume_path = getattr(self.opt.training, 'pretrained_ckpt', None)

        is_resuming = resume_path is not None and os.path.exists(resume_path)
        if is_resuming:
            epoch_done = self.load(resume_path, self.device)
            torch.cuda.empty_cache()
        else:
            if os.path.exists(self.opt.training.pretrained_w):
                w_dict = torch.load(self.opt.training.pretrained_w, map_location='cpu', weights_only=False)
                self.models.W.load_state_dict(w_dict['WriterIdentifier'], strict=False)
                self.models.B.load_state_dict(w_dict['StyleBackbone'], strict=False)
                self.print(f'load pretrained writer_identifier: {self.opt.training.pretrained_w}')
                # self.validate_wid()
            if os.path.exists(self.opt.training.pretrained_r):
                r_dict = torch.load(self.opt.training.pretrained_r, map_location='cpu', weights_only=False)['Recognizer']
                self.models.R.load_state_dict(r_dict, strict=False)
                self.print(f'load pretrained recognizer: {self.opt.training.pretrained_r}')
                # self.validate_ocr()

        restored_meta = getattr(self, 'restored_metadata', {})
        restored_iter = restored_meta.get('iter_count', None)
        restored_ema_step = restored_meta.get('ema_step', None)

        if restored_iter is not None:
            iter_count = restored_iter + 1
            self.print(f"Resumed exact iter_count={iter_count} from checkpoint")
            start_epoch = iter_count // len(self.train_loader) + 1
            skip_batches = iter_count % len(self.train_loader)
        elif is_resuming:
            start_epoch = epoch_done + 1
            skip_batches = 0
            iter_count = epoch_done * len(self.train_loader)
            self.print(f"Calculated iter_count={iter_count} based on epoch_done={epoch_done}")
        else:
            start_epoch = 1
            skip_batches = 0
            iter_count = 0

        self.epoch_start = start_epoch

        self.lr_schedulers = Munch(
            G=get_scheduler(self.optimizers.G, opt.training, last_epoch=start_epoch - 2 if is_resuming else -1),
            D=get_scheduler(self.optimizers.D, opt.training, last_epoch=start_epoch - 2 if is_resuming else -1)
        )
        if hasattr(self, '_ckpt_sched_data') and self._ckpt_sched_data:
            for key in self.lr_schedulers.keys():
                sched_key = 'SCHED.' + key
                if sched_key in self._ckpt_sched_data:
                    try:
                        self.lr_schedulers[key].load_state_dict(self._ckpt_sched_data[sched_key])
                        if hasattr(self.lr_schedulers[key], 'get_last_lr') and key in self.optimizers:
                            lrs = self.lr_schedulers[key].get_last_lr()
                            for param_group, lr in zip(self.optimizers[key].param_groups, lrs):
                                param_group['lr'] = lr
                        self.print(f'Loaded restored scheduler state for {key}')
                    except Exception as e:
                        self.print(f'Failed to restore scheduler state for {key}: {e}')

        # multi-gpu
        if self.local_rank > -1:
            for key in self.models.keys():
                self.models[key] = torch.nn.parallel.DistributedDataParallel(
                    self.models[key],
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False
                )

        self.averager_meters = AverageMeterManager(['adv_loss', 'fake_disc_loss',
                                                    'real_disc_loss', 'adv_loss_patch',
                                                    'fake_disc_loss_patch',
                                                    'real_disc_loss_patch', 'recn_loss',
                                                    'fake_ctc_loss', 'info_loss',
                                                    'style_contrastive_loss',
                                                    'fake_wid_loss', 'ctx_loss',
                                                    'kl_loss', 'gram_loss', 'gp_ctc', 'gp_info',
                                                    'gp_wid', 'gp_recn'])
        device = self.device

        ctc_len_scale = self.unwrap_model(self.models.R).len_scale

        best_fid = restored_meta.get('best_fid', None)
        if best_fid is None:
            best_fid = np.inf
        else:
            self.print(f"Resumed best_fid={best_fid:.4f} from checkpoint")

        if self.use_ema:
            if restored_ema_step is not None:
                self.ema_tracker.step = restored_ema_step
                self.print(f"Restored EMA tracker step={self.ema_tracker.step} from checkpoint")
            else:
                self.ema_tracker.step = iter_count // opt.training.num_critic_train
                self.print(f"Set EMA tracker step to {self.ema_tracker.step} based on iter_count={iter_count}")
        is_best = False
        best_scores = None

        _should_restore_rng = is_resuming and skip_batches > 0
        for epoch in range(start_epoch, self.opt.training.epochs + 1):
            if getattr(self, 'train_sampler', None) is not None:
                self.train_sampler.set_epoch(epoch)
            for i, batch in enumerate(self.train_loader):
                if epoch == start_epoch and i < skip_batches:
                    continue
                
                if _should_restore_rng:
                    self.restore_rng_state()
                    _should_restore_rng = False
                #############################
                # Prepare inputs & Network Forward
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = batch['style_imgs'].to(device, non_blocking=True), \
                                                      batch['style_img_lens'].to(device, non_blocking=True), \
                                                      batch['wids'].to(device, non_blocking=True)
                real_aug_imgs, real_aug_img_lens = batch['aug_imgs'].to(device, non_blocking=True), batch['aug_img_lens'].to(device, non_blocking=True)
                real_lbs, real_lb_lens = batch['lbs'].to(device, non_blocking=True), batch['lb_lens'].to(device, non_blocking=True)
                max_label_len = real_lbs.size(-1)

                #############################
                # Optimizing Discriminator
                #############################
                self.optimizers.D.zero_grad(set_to_none=True)
                set_requires_grad([self.models.G, self.models.E, self.models.R, self.models.W, self.models.B], False)
                set_requires_grad([self.models.D, self.models.P], True)
                # self.models.B.frozen_bn()

                with torch.no_grad():
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                 self.opt.training.capitalize_ratio,
                                                 self.opt.training.blank_ratio)
                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    z_in = self.z

                    if self.vae_mode:
                        enc_z, _, _ = self.models.E(real_imgs, real_img_lens, self.models.B, vae_mode=True)
                    else:
                        enc_z = self.models.E(real_imgs, real_img_lens, self.models.B, vae_mode=False)

                    # Batch forward all fake/generated types to avoid multiple GPU kernel launches
                    cat_z = torch.cat([z_in, enc_z, enc_z], dim=0)
                    cat_fake_lb_lens = torch.cat([fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0)
                    cat_y = torch.cat([fake_lbs, fake_lbs, real_lbs], dim=0)
                    cat_fake_imgs = self.models.G(cat_z, cat_y, cat_fake_lb_lens)
                    fake_imgs, style_imgs, recn_imgs = torch.chunk(cat_fake_imgs, 3, dim=0)
                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width

                ### Compute discriminative loss for real & fake samples ###
                # Refactored to avoid torch.cat and save memory
                fake_img_lens = fake_lb_lens * self.opt.char_width
                style_img_lens = fake_lb_lens * self.opt.char_width
                recn_img_lens = real_lb_lens * self.opt.char_width
                
                # Batch forward all generated types through D to avoid multiple GPU kernel launches
                d_fake_all = self.models.D(cat_fake_imgs.detach(), cat_fake_img_lens, cat_fake_lb_lens)
                d_fake, d_style, d_recn = torch.chunk(d_fake_all, 3, dim=0)
                fake_disc_loss = (torch.mean(F.relu(1.0 + d_fake)) + 
                                  torch.mean(F.relu(1.0 + d_style)) + 
                                  torch.mean(F.relu(1.0 + d_recn))) / 3

                # Patch Discriminator forwards
                n_patch_row = (cat_fake_imgs.size(-2) - 32) // 8 + 1
                n_fake = int(torch.sum(torch.div(fake_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row
                n_style = int(torch.sum(torch.div(style_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row
                n_recn = int(torch.sum(torch.div(recn_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row

                p_all_patches = extract_all_patches(cat_fake_imgs.detach(), cat_fake_img_lens)
                masking_mode = getattr(self.opt.training, 'masking_mode', 'none')
                if masking_mode != 'none':
                    p_all_patches = apply_light_mixed_patch_mask(p_all_patches)

                # Batch forward all patches through P to avoid multiple GPU kernel launches
                p_all = self.models.P(p_all_patches)
                p_fake, p_style, p_recn = torch.split(p_all, [n_fake, n_style, n_recn], dim=0)
                fake_disc_loss_patch = (torch.mean(F.relu(1.0 + p_fake)) + 
                                        torch.mean(F.relu(1.0 + p_style)) + 
                                        torch.mean(F.relu(1.0 + p_recn))) / 3

                # real_imgs.requires_grad_()
                real_disc = self.models.D(real_imgs, real_img_lens, real_lb_lens)
                real_aug_lb_lens = real_lb_lens * (real_aug_img_lens.float() / torch.clamp(real_img_lens.float(), min=1.0))
                real_disc_aug = self.models.D(real_aug_imgs, real_aug_img_lens, real_aug_lb_lens)
                real_disc_loss = (torch.mean(F.relu(1.0 - real_disc)) +
                                  torch.mean(F.relu(1.0 - real_disc_aug))) / 2

                real_img_patches = extract_all_patches(real_imgs, real_img_lens, plot=False)
                real_aug_imgs_patches = extract_all_patches(real_aug_imgs, real_aug_img_lens)
                real_patches_cat = torch.cat([real_img_patches, real_aug_imgs_patches], dim=0)
                if masking_mode != 'none':
                    real_patches_cat = apply_light_mixed_patch_mask(real_patches_cat)
                real_disc_patches = self.models.P(real_patches_cat)
                real_disc_loss_patch = torch.mean(F.relu(1.0 - real_disc_patches))

                disc_loss = real_disc_loss + fake_disc_loss + real_disc_loss_patch + fake_disc_loss_patch
                self.averager_meters.update('real_disc_loss', real_disc_loss.item())
                self.averager_meters.update('fake_disc_loss', fake_disc_loss.item())
                self.averager_meters.update('real_disc_loss_patch', real_disc_loss_patch.item())
                self.averager_meters.update('fake_disc_loss_patch', fake_disc_loss_patch.item())

                disc_loss.backward()
                self.optimizers.D.step()

                #############################
                # Optimizing Generator
                #############################
                if iter_count % self.opt.training.num_critic_train == 0:
                    self.optimizers.G.zero_grad(set_to_none=True)
                    set_requires_grad([self.models.D, self.models.P, self.models.R, self.models.W, self.models.B], False)
                    set_requires_grad([self.models.G, self.models.E], True)
                    # self.models.B.frozen_bn()

                    ##########################
                    # Prepare Fake Inputs
                    ##########################
                    self.y.sample_()
                    sampled_words = idx_to_words(self.y, self.lexicon, max_label_len,
                                                 self.opt.training.capitalize_ratio,
                                                 self.opt.training.blank_ratio,
                                                 sort=True)

                    fake_lbs, fake_lb_lens = self.label_converter.encode(sampled_words, max_label_len)
                    fake_lbs, fake_lb_lens = fake_lbs.to(device).detach(), fake_lb_lens.to(device).detach()

                    self.z.sample_()
                    z_in = self.z

                    # Keep style encoder inputs clean as masking is applied strictly to local patches
                    masking_mode = getattr(self.opt.training, 'masking_mode', 'none')
                    if self.vae_mode:
                        (enc_z, mu, logvar), real_img_feats = self.models.E(real_imgs, real_img_lens, self.models.B,
                                                                            ret_feats=True, vae_mode=True)
                    else:
                        enc_z, real_img_feats = self.models.E(real_imgs, real_img_lens, self.models.B,
                                                              ret_feats=True, vae_mode=False)

                    # Batch forward all fake/generated types through G to avoid multiple GPU kernel launches
                    cat_z = torch.cat([z_in, enc_z, enc_z], dim=0)
                    cat_fake_lb_lens = torch.cat([fake_lb_lens, fake_lb_lens, real_lb_lens], dim=0)
                    cat_y = torch.cat([fake_lbs, fake_lbs, real_lbs], dim=0)
                    cat_fake_imgs = self.models.G(cat_z, cat_y, cat_fake_lb_lens)
                    fake_imgs, style_imgs, recn_imgs = torch.chunk(cat_fake_imgs, 3, dim=0)

                    ###################################################
                    # Calculating G Losses
                    ####################################################
                    ### deal with fake samples ###
                    ### Compute Adversarial loss ###
                    # Refactored to avoid torch.cat and save memory
                    fake_img_lens = fake_lb_lens * self.opt.char_width
                    style_img_lens = fake_lb_lens * self.opt.char_width
                    recn_img_lens = real_lb_lens * self.opt.char_width

                    cat_fake_img_lens = cat_fake_lb_lens * self.opt.char_width
                    # Batch forward all generated types through D to avoid multiple GPU kernel launches
                    d_fake_all = self.models.D(cat_fake_imgs, cat_fake_img_lens, cat_fake_lb_lens)
                    d_fake, d_style, d_recn = torch.chunk(d_fake_all, 3, dim=0)
                    adv_loss = -(torch.mean(d_fake) + torch.mean(d_style) + torch.mean(d_recn)) / 3

                    n_patch_row = (cat_fake_imgs.size(-2) - 32) // 8 + 1
                    n_fake = int(torch.sum(torch.div(fake_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row
                    n_style = int(torch.sum(torch.div(style_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row
                    n_recn = int(torch.sum(torch.div(recn_img_lens - 32, 8, rounding_mode='trunc') + 1).item()) * n_patch_row

                    p_all_patches = extract_all_patches(cat_fake_imgs, cat_fake_img_lens)
                    if masking_mode != 'none':
                        p_all_patches = apply_light_mixed_patch_mask(p_all_patches)

                    # Batch forward all patches through P to avoid multiple GPU kernel launches
                    p_all = self.models.P(p_all_patches)
                    p_fake, p_style, p_recn = torch.split(p_all, [n_fake, n_style, n_recn], dim=0)
                    adv_loss_patch = -(torch.mean(p_fake) + torch.mean(p_style) + torch.mean(p_recn)) / 3

                    ### CTC Auxiliary loss ###
                    # self.models.R.frozen_bn()
                    fake_img_lens = fake_lb_lens * self.opt.char_width
                    style_img_lens = fake_lb_lens * self.opt.char_width
                    recn_img_lens = real_lb_lens * self.opt.char_width

                    # Batch forward all generated types through R to avoid multiple GPU kernel launches
                    cat_fake_ctc = self.models.R(cat_fake_imgs, cat_fake_img_lens)
                    fake_ctc_rand, fake_ctc_style, fake_ctc_recn = torch.chunk(cat_fake_ctc, 3, dim=1)

                    fake_ctc_loss_rand = self.ctc_loss(fake_ctc_rand, fake_lbs,
                                                       torch.div(fake_img_lens, ctc_len_scale, rounding_mode='trunc'),
                                                       fake_lb_lens)

                    fake_ctc_loss_style = self.ctc_loss(fake_ctc_style, fake_lbs,
                                                        torch.div(style_img_lens, ctc_len_scale, rounding_mode='trunc'),
                                                        fake_lb_lens)

                    fake_ctc_loss_recn = self.ctc_loss(fake_ctc_recn, real_lbs,
                                                       torch.div(recn_img_lens, ctc_len_scale, rounding_mode='trunc'),
                                                       real_lb_lens)

                    fake_ctc_loss = fake_ctc_loss_rand + fake_ctc_loss_recn + fake_ctc_loss_style

                    ### Style Reconstruction & Contrastive Loss ###
                    styles = self.models.E(fake_imgs, fake_lb_lens * self.opt.char_width, self.models.B)
                    info_loss = torch.mean(torch.abs(styles - z_in.detach()))
                    style_contrastive_loss = contrastive_style_loss(styles, z_in.detach())

                    ### Content Restruction ###
                    recn_loss = recn_l1_loss(recn_imgs, real_imgs, real_img_lens)

                    ### Writer Identify Loss ###
                    cat_style_imgs = torch.cat([style_imgs, recn_imgs], dim=0)
                    cat_style_img_lens = torch.cat([fake_lb_lens, real_lb_lens], dim=0) * self.opt.char_width
                    recn_wid_logits, fake_imgs_feats = self.models.W(cat_style_imgs,
                                                                     cat_style_img_lens,
                                                                     self.models.B,
                                                                     ret_feats=True)
                    fake_wid_loss = self.classify_loss(recn_wid_logits, real_wids.repeat(2))

                    ###  Contextual Loss and Gram Loss for non-aligned data  ###
                    ctx_loss = torch.tensor(0.0, device=self.device)
                    gram_loss = torch.tensor(0.0, device=self.device)
                    for real_img_feat, fake_img_feat \
                            in zip(real_img_feats, fake_imgs_feats):
                        fake_feat = fake_img_feat.chunk(2, dim=0)
                        # ctx_loss for style_imgs
                        ctx_loss += self.contextual_loss(real_img_feat, fake_feat[0])
                        # ctx_loss for recn_imgs
                        ctx_loss += self.contextual_loss(real_img_feat, fake_feat[1])

                        # gram_loss
                        gram_loss += self.gram_loss(fake_feat[0], real_img_feat)
                        gram_loss += self.gram_loss(fake_feat[1], real_img_feat)

                    kl_loss = KLloss(mu, logvar) if self.vae_mode else torch.tensor(0.0, device=self.device)

                    grad_cat_fake_adv = torch.autograd.grad(adv_loss, cat_fake_imgs, create_graph=False, retain_graph=True)[0]
                    grad_fake_adv = torch.chunk(grad_cat_fake_adv, 3, dim=0)[0]
                    grad_fake_OCR = torch.autograd.grad(fake_ctc_loss_rand, fake_ctc_rand, create_graph=False, retain_graph=True)[0]
                    grad_fake_info = torch.autograd.grad(info_loss, fake_imgs, create_graph=False, retain_graph=True)[0]
                    grad_fake_wid = torch.autograd.grad(fake_wid_loss, recn_wid_logits, create_graph=False, retain_graph=True)[0]
                    grad_fake_recn = torch.autograd.grad(recn_loss, enc_z, create_graph=False, retain_graph=True)[0]

                    std_grad_adv = torch.std(grad_fake_adv)
                    gp_ctc = torch.div(std_grad_adv, torch.std(grad_fake_OCR) + 1e-8).detach() + 1
                    gp_ctc.clamp_max_(100)
                    gp_ctc = gp_ctc * getattr(self.opt.training, 'lambda_ctc', 1.0)
                    gp_info = torch.div(std_grad_adv, torch.std(grad_fake_info) + 1e-8).detach() + 1
                    gp_wid = torch.div(std_grad_adv, torch.std(grad_fake_wid) + 1e-8).detach() + 1
                    gp_wid.clamp_max_(10)
                    gp_recn = torch.div(std_grad_adv, torch.std(grad_fake_recn) + 1e-8).detach() + 1

                    self.averager_meters.update('gp_ctc', gp_ctc.item())
                    self.averager_meters.update('gp_info', gp_info.item())
                    self.averager_meters.update('gp_wid', gp_wid.item())
                    self.averager_meters.update('gp_recn', gp_recn.item())

                    g_loss = adv_loss + adv_loss_patch +\
                             gp_ctc * fake_ctc_loss + \
                             gp_info * (info_loss + style_contrastive_loss) + \
                             gp_wid * fake_wid_loss + \
                             gp_recn * recn_loss + \
                             self.opt.training.lambda_ctx * ctx_loss + \
                             self.opt.training.lambda_gram * gram_loss + \
                             self.opt.training.lambda_kl * kl_loss
                    
                    g_loss.backward()
                    self.averager_meters.update('adv_loss', adv_loss.item())
                    self.averager_meters.update('adv_loss_patch', adv_loss_patch.item())
                    self.averager_meters.update('fake_ctc_loss', fake_ctc_loss.item())
                    self.averager_meters.update('info_loss', info_loss.item())
                    self.averager_meters.update('style_contrastive_loss', style_contrastive_loss.item())
                    self.averager_meters.update('fake_wid_loss', fake_wid_loss.item())
                    self.averager_meters.update('recn_loss', recn_loss.item())
                    self.averager_meters.update('ctx_loss', ctx_loss.item())
                    self.averager_meters.update('gram_loss', gram_loss.item())
                    self.averager_meters.update('kl_loss', kl_loss.item())
                    self.optimizers.G.step()
                    if self.use_ema:
                        self.ema_tracker.step_ema(self.models_ema.G, self.models.G)
                        self.ema_tracker.step_ema(self.models_ema.E, self.models.E)
                        self.ema_tracker.step_ema(self.models_ema.B, self.models.B)
                        self.ema_tracker.step += 1

                if iter_count % self.opt.training.print_iter_val == 0:
                    meter_vals = self.averager_meters.eval_all()
                    self.averager_meters.reset_all()
                    info = "[%3d|%3d]-[%4d|%4d] G:%.4f G-p:%.4f D-fake:%.4f D-real:%.4f " \
                           "D-fake-p:%.4f D-real-p:%.4f CTC-fake:%.4f Wid-fake:%.4f " \
                           "Recn-z:%.4f Cont-z:%.4f Recn-c:%.4f Ctx:%.4f Gram:%.4f Kl:%.4f" \
                           % (epoch, self.opt.training.epochs,
                              iter_count % len(self.train_loader), len(self.train_loader),
                              meter_vals['adv_loss'], meter_vals['adv_loss_patch'],
                              meter_vals['fake_disc_loss'], meter_vals['real_disc_loss'],
                              meter_vals['fake_disc_loss_patch'], meter_vals['real_disc_loss_patch'],
                              meter_vals['fake_ctc_loss'], meter_vals['fake_wid_loss'], meter_vals['info_loss'],
                              meter_vals['style_contrastive_loss'], meter_vals['recn_loss'], meter_vals['ctx_loss'],
                              meter_vals['gram_loss'], meter_vals['kl_loss'])
                    self.print(info) if self.local_rank < 1 else None

                    if _is_master:


                        # WandB losses
                        wandb_log = {('loss/' + k): v for k, v in meter_vals.items()}
                        wandb_log['train/iter'] = iter_count + 1

                        try:
                            lr = self.lr_schedulers.G.get_last_lr()[0]
                        except Exception:
                            lr = self.lr_schedulers.G.get_lr()[0]
                        wandb_log['loss/lr'] = lr

                        G_unwrapped = getattr(self.models.G, 'module', self.models.G)
                        info_attns = G_unwrapped._info_attention()
                        for i_, attn_info in enumerate(info_attns):
                            wandb_log['loss/gamma%d' % i_] = attn_info['gamma']

                        import wandb as _wandb
                        if _wandb.run:
                            _wandb.log(wandb_log, step=iter_count + 1)

                if (iter_count + 1) % self.opt.training.sample_iter_val == 0:
                    if not self.logger:
                        self.create_logger() if self.local_rank < 1 else None

                    sample_root = os.path.join(self.log_root, self.opt.training.sample_dir)
                    if not os.path.exists(sample_root):
                        os.makedirs(sample_root) if self.local_rank < 1 else None
                    self.sample_images(iter_count + 1) if self.local_rank < 1 else None

                eval_epoch_val = self.opt.training.get('eval_epoch_val', 0.5)
                save_epoch_val = self.opt.training.get('save_epoch_val', 1.0)
                
                eval_interval_iters = max(1, int(eval_epoch_val * len(self.train_loader)))
                save_interval_iters = max(1, int(save_epoch_val * len(self.train_loader)))
                
                global_iter = (epoch - 1) * len(self.train_loader) + (i + 1)
                is_eval = global_iter % eval_interval_iters == 0
                is_save = global_iter % save_interval_iters == 0
                
                if getattr(self, 'is_resumed_start', False):
                    is_eval = False
                    self.is_resumed_start = False

                if is_eval:
                    self.print('Calculate FID_KID (iter {})'.format(iter_count + 1)) if self.local_rank < 1 else None
                    scores = self.validate(current_epoch=epoch)
                    if 'fid' in scores:
                        self.last_eval_fid = float(scores['fid'])
                    if _is_master:
                        score_str = ", ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}" for k, v in scores.items()])
                        self.print(f"Validation metrics at iter {iter_count + 1}: {score_str}")
                        import wandb as _wandb
                        if _wandb.run:
                            _wandb.log({'valid/' + k: v for k, v in scores.items()}, step=iter_count + 1)

                    if 'fid' in scores and scores['fid'] < best_fid:
                        best_fid = scores['fid']
                        best_scores = scores
                        if _is_master:
                            self.save('best', epoch, iter_count=iter_count, best_fid=best_fid, **(best_scores or {}))

                if is_save:
                    if _is_master:
                        current_eval_fid = float(scores['fid']) if (is_eval and 'scores' in locals() and isinstance(scores, dict) and 'fid' in scores) else getattr(self, 'last_eval_fid', None)
                        if current_eval_fid is None and hasattr(self, 'restored_metadata'):
                            current_eval_fid = self.restored_metadata.get('last_eval_fid', self.restored_metadata.get('best_fid', None))
                        self.save('last', epoch, iter_count=iter_count, best_fid=best_fid, fid=current_eval_fid)

                iter_count += 1
                if getattr(self, 'is_resumed_start', False):
                    self.is_resumed_start = False

            if epoch:
                if self.local_rank > -1:
                    dist.barrier()

            for scheduler in self.lr_schedulers.values():
                scheduler.step()

        if _is_master:
            import wandb as _wandb
            _wandb.finish()


class RecognizeModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(RecognizeModel, self).__init__(opt, log_root)

        device = self.device
        self.collect_fn = get_collect_fn(sort_input=opt.training.sort_input, sort_style=False)
        recognizer = Recognizer(**opt.OcrModel).to(device)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device, weights_only=False)['Recognizer']
            new_ckpt = {}
            for key, val in ckpt.items():
                if not key.startswith('ctc_cls'):
                    new_ckpt[key] = val
            recognizer.load_state_dict(new_ckpt, strict=False)
            self.print(f'load pretrained backbone from {opt.training.pretrained_backbone}')

        self.models = Munch(R=recognizer)

        self.tst_loader = DataLoader(
            get_dataset(self.opt.valid.dset_name, self.opt.valid.dset_split, process_style=True),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=True, sort_style=True)
        )

        self.ctc_loss = CTCLoss(zero_infinity=True, reduction='mean')

    def train(self):
        self.info()

        trainset_info = (self.opt.training.dset_name, self.opt.training.dset_split, False, self.opt.training.augment, True)
        self.print('Trainset: {} [{}]'.format(*trainset_info))
        trainset = get_dataset(*trainset_info)
        if self.local_rank > -1:
            from torch.utils.data.distributed import DistributedSampler
            self.train_sampler = DistributedSampler(trainset, num_replicas=None, rank=self.local_rank, shuffle=True)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        self.train_loader = DataLoader(
            trainset,
            batch_size=self.opt.training.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            collate_fn=self.collect_fn,
            num_workers=4,
            pin_memory=(self.device.type == 'cuda'),
            worker_init_fn=seed_worker
        )

        if self.local_rank > -1:
            self.models.R = torch.nn.parallel.DistributedDataParallel(
                self.models.R,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False
            )

        self.optimizers = Munch(R=torch.optim.Adam(self.models.R.parameters(), lr=self.opt.training.lr))

        epoch_done = 1
        is_resuming = bool(self.opt.training.resume and (os.path.exists(str(self.opt.training.resume)) or self.resolve_resume_path(self.opt.training.resume)))
        if is_resuming:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        restored_meta = getattr(self, 'restored_metadata', {})
        restored_iter = restored_meta.get('iter_count', None)
        if restored_iter is not None:
            iter_count = restored_iter + 1
            start_epoch = iter_count // len(self.train_loader) + 1
            skip_batches = iter_count % len(self.train_loader)
        elif is_resuming:
            start_epoch = epoch_done + 1
            skip_batches = 0
            iter_count = epoch_done * len(self.train_loader)
        else:
            start_epoch = 1
            skip_batches = 0
            iter_count = 0

        self.lr_schedulers = Munch(R=get_scheduler(self.optimizers.R, self.opt.training, last_epoch=start_epoch - 2 if is_resuming else -1))
        if hasattr(self, '_ckpt_sched_data') and self._ckpt_sched_data and 'SCHED.R' in self._ckpt_sched_data:
            try:
                self.lr_schedulers.R.load_state_dict(self._ckpt_sched_data['SCHED.R'])
                if hasattr(self.lr_schedulers.R, 'get_last_lr') and 'R' in self.optimizers:
                    lrs = self.lr_schedulers.R.get_last_lr()
                    for param_group, lr in zip(self.optimizers.R.param_groups, lrs):
                        param_group['lr'] = lr
            except Exception:
                pass

        device = self.device
        ctc_loss_meter = AverageMeter()
        recognizer_unwrapped = self.unwrap_model(self.models.R)
        ctc_len_scale = recognizer_unwrapped.len_scale
        best_cer = np.inf

        for epoch in range(start_epoch, self.opt.training.epochs + 1):
            if getattr(self, 'train_sampler', None) is not None:
                self.train_sampler.set_epoch(epoch)
            for i, batch in enumerate(self.train_loader):
                if epoch == start_epoch and i < skip_batches:
                    continue
                #############################
                # Prepare inputs
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens = batch['aug_imgs'].to(device), batch['aug_img_lens'].to(device)
                real_lbs, real_lb_lens = batch['lbs'].to(device), batch['lb_lens'].to(device)

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.R.zero_grad(set_to_none=True)
                ### Compute CTC loss for real samples###
                real_ctc = self.models.R(real_imgs, real_img_lens)
                real_ctc_lens = real_img_lens // ctc_len_scale
                real_ctc_loss = self.ctc_loss(real_ctc, real_lbs, real_ctc_lens, real_lb_lens)
                ctc_loss_meter.update(real_ctc_loss.item())
                real_ctc_loss.backward()
                self.optimizers.R.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not self.logger:
                            self.create_logger()

                    try:
                        lr = self.lr_schedulers.R.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.R.get_lr()[0]

                    ctc_loss_avg = ctc_loss_meter.eval()
                    ctc_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] CTC: %.5f  Lr: %.6f" \
                           % (epoch, self.opt.training.epochs, iter_count % len(self.train_loader),
                              len(self.train_loader), ctc_loss_avg, lr)
                    self.print(info)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root) if self.local_rank < 1 else None

                self.save('last', epoch, iter_count=iter_count)
                if self.local_rank > -1:
                    dist.barrier()


            for scheduler in self.lr_schedulers.values():
                scheduler.step()

    def validate(self, *args, **kwargs):
        self.set_mode('eval')
        ctc_len_scale = self.unwrap_model(self.models.R).len_scale
        char_trans = 0
        total_chars = 0
        word_trans = 0
        total_words = 0
        self.print(self.tst_loader.dataset.file_path)
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.tst_loader), total=len(self.tst_loader)):
                real_imgs, real_img_lens = batch['style_imgs'].to(self.device), batch['style_img_lens'].to(self.device)
                logits = self.models.R(real_imgs, real_img_lens)
                logits = torch.nn.functional.softmax(logits, dim=2).detach()

                logits = logits.cpu().numpy()
                word_preds = []
                for logit, img_len in zip(logits, batch['style_img_lens'].cpu().numpy()):
                    label = ctc_greedy_decoder(logit[:img_len // ctc_len_scale])
                    word_preds.append(self.label_converter.decode(label))

                word_reals = self.label_converter.decode(batch['lbs'], batch['lb_lens'])

                for word_pred, word_real in zip(word_preds, word_reals):
                    char_tran = levenshtein(word_pred, word_real)
                    char_trans += char_tran
                    total_chars += len(word_real)
                    total_words += 1
                    if char_tran > 0:
                        word_trans += 1

        for model in self.models.values():
            model.train()

        cer = char_trans * 1.0 / max(total_chars, 1)
        wer = word_trans * 1.0 / max(total_words, 1)
        return {'CER': cer, 'WER': wer}


class WriterIdentifyModel(BaseModel):
    def __init__(self, opt, log_root='./'):
        super(WriterIdentifyModel, self).__init__(opt, log_root)

        device = self.device

        style_backbone = StyleBackbone(**opt.StyBackbone).to(device)
        if os.path.exists(opt.training.pretrained_backbone):
            ckpt = torch.load(opt.training.pretrained_backbone, device, weights_only=False)

            if 'Recognizer' in ckpt:
                ckpt = ckpt['Recognizer']
                new_ckpt = {}
                for key, val in ckpt.items():
                    if key.startswith('cnn_backbone') or key.startswith('cnn_ctc'):
                        new_ckpt[key] = val
                style_backbone.load_state_dict(new_ckpt)
            else:
                ckpt = ckpt['StyleBackbone']
                style_backbone.load_state_dict(ckpt)

            self.print(f'Load style_backbone from {opt.training.pretrained_backbone}')

        identifier = WriterIdentifier(**opt.WidModel).to(device)
        self.models = Munch(W=identifier, B=style_backbone)

        self.tst_loader = DataLoader(
            get_dataset(opt.dataset, opt.valid.dset_split),
            batch_size=opt.valid.batch_size,
            shuffle=False,
            collate_fn=get_collect_fn(sort_input=False)
        )

        self.wid_loss = CrossEntropyLoss()

    def train(self):
        self.info()

        trainset_info = (self.opt.training.dset_name,
                         self.opt.training.dset_split,
                         self.opt.training.random_clip,
                         False, self.opt.training.process_style)
        self.print('Trainset: {} [{}]'.format(*trainset_info))
        trainset = get_dataset(*trainset_info)
        if self.local_rank > -1:
            from torch.utils.data.distributed import DistributedSampler
            self.train_sampler = DistributedSampler(trainset, num_replicas=None, rank=self.local_rank, shuffle=True)
            shuffle = False
        else:
            self.train_sampler = None
            shuffle = True

        self.train_loader = DataLoader(
            trainset,
            batch_size=self.opt.training.batch_size,
            shuffle=shuffle,
            sampler=self.train_sampler,
            collate_fn=get_collect_fn(sort_input=True, sort_style=False),
            num_workers=4,
            pin_memory=(self.device.type == 'cuda'),
            worker_init_fn=seed_worker
        )

        if self.local_rank > -1:
            for key in self.models.keys():
                self.models[key] = torch.nn.parallel.DistributedDataParallel(
                    self.models[key],
                    device_ids=[self.local_rank],
                    output_device=self.local_rank,
                    broadcast_buffers=False
                )

        if self.opt.training.frozen_backbone:
            self.print('frozen_backbone')
            self.optimizers = Munch(W=torch.optim.Adam(self.models.W.parameters(), lr=self.opt.training.lr))
        else:
            self.optimizers = Munch(W=torch.optim.Adam(
                                        chain(self.models.W.parameters(), self.models.B.parameters()),
                                    lr=self.opt.training.lr))

        epoch_done = 1
        is_resuming = bool(self.opt.training.resume and os.path.exists(self.opt.training.resume))
        if is_resuming:
            epoch_done = self.load(self.opt.training.resume)
            self.print(self.validate())

        restored_meta = getattr(self, 'restored_metadata', {})
        restored_iter = restored_meta.get('iter_count', None)
        if restored_iter is not None:
            iter_count = restored_iter + 1
            start_epoch = iter_count // len(self.train_loader) + 1
            skip_batches = iter_count % len(self.train_loader)
        elif is_resuming:
            start_epoch = epoch_done + 1
            skip_batches = 0
            iter_count = epoch_done * len(self.train_loader)
        else:
            start_epoch = 1
            skip_batches = 0
            iter_count = 0

        self.lr_schedulers = Munch(W=get_scheduler(self.optimizers.W, self.opt.training, last_epoch=start_epoch - 2 if is_resuming else -1))
        if hasattr(self, '_ckpt_sched_data') and self._ckpt_sched_data and 'SCHED.W' in self._ckpt_sched_data:
            try:
                self.lr_schedulers.W.load_state_dict(self._ckpt_sched_data['SCHED.W'])
            except Exception:
                pass

        device = self.device
        wid_loss_meter = AverageMeter()
        best_wrr = 0

        for epoch in range(start_epoch, self.opt.training.epochs + 1):
            if getattr(self, 'train_sampler', None) is not None:
                self.train_sampler.set_epoch(epoch)
            for i, batch in enumerate(self.train_loader):
                if epoch == start_epoch and i < skip_batches:
                    continue
                #############################
                # Prepare inputs
                #############################
                self.set_mode('train')
                real_imgs, real_img_lens, real_wids = batch['aug_imgs'].to(device), \
                                                      batch['aug_img_lens'].to(device), \
                                                      batch['wids'].to(device)

                if self.opt.training.frozen_backbone:
                    b_module = self.unwrap_model(self.models.B)
                    frozen_bn(b_module)

                #############################
                # OptimizingRecognizer
                #############################
                self.optimizers.W.zero_grad(set_to_none=True)
                ### Compute CTC loss for real samples###
                wid_logits = self.models.W(real_imgs, real_img_lens, self.models.B)
                wid_loss = self.wid_loss(wid_logits, real_wids)
                wid_loss_meter.update(wid_loss.item())
                wid_loss.backward()
                self.optimizers.W.step()

                if iter_count % self.opt.training.print_iter_val == 0:
                    if epoch > 1 and not self.logger:
                            self.create_logger()

                    try:
                        lr = self.lr_schedulers.W.get_last_lr()[0]
                    except Exception:
                        lr = self.lr_schedulers.W.get_lr()[0]

                    wid_loss_avg = wid_loss_meter.eval()
                    wid_loss_meter.reset()
                    info = "[%3d|%3d]-[%4d|%4d] WID: %.5f  Lr: %.6f" \
                           % (epoch, self.opt.training.epochs, iter_count % len(self.train_loader),
                              len(self.train_loader), wid_loss_avg, lr)
                    self.print(info)

                iter_count += 1

            if epoch:
                ckpt_root = os.path.join(self.log_root, self.opt.training.ckpt_dir)
                if not os.path.exists(ckpt_root):
                    os.makedirs(ckpt_root) if self.local_rank < 1 else None

                self.save('last', epoch, iter_count=iter_count)
                if self.local_rank > -1:
                    dist.barrier()


            for scheduler in self.lr_schedulers.values():
                scheduler.step()

    def validate(self, *args, **kwargs):
        self.set_mode('eval')

        with torch.no_grad():
            acc_counts = 0.
            total_counts = 0.
            for i, batch in tqdm(enumerate(self.tst_loader), total=len(self.tst_loader)):
                wid_logits = self.models.W(batch['style_imgs'].to(self.device),
                                           batch['style_img_lens'].to(self.device),
                                           self.models.B)
                _, preds = torch.max(wid_logits.data, dim=1)

                acc_counts += preds.eq(batch['wids'].to(self.device)).sum().item()
                total_counts += wid_logits.size(0)

            wrr = acc_counts * 100. / total_counts
            wier = 1 - acc_counts * 1. / total_counts
            self.print(f'wier: {wier}')

        for model in self.models.values():
            model.train()

        return wrr