"""Native64 HiGAN+, deterministic encoder mean and explicit downstream conversion."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.external import GeneratorAdapter, adapter_cli


def load_adapter(model_cfg, device='cpu'):
    return GeneratorAdapter(model_cfg, device, 'higanplus')


if __name__ == '__main__':
    adapter_cli('higanplus')
