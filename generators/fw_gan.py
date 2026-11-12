"""Native32 FW-GAN, generator/encoder/shared-backbone only."""
if __package__ in (None, ''):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generators.external import GeneratorAdapter, adapter_cli


def load_adapter(model_cfg, device='cpu'):
    return GeneratorAdapter(model_cfg, device, 'fw_gan')


if __name__ == '__main__':
    adapter_cli('fw_gan')
