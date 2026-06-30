import argparse
from importlib import util as importlib_util
from importlib.metadata import PackageNotFoundError, version
import re
import subprocess
import sys
from pathlib import Path


BACKBONE_CONFIGS = {
    'vit_small': 'projects/configs/co_dino_vit/co_dino_5scale_vit_small_coco_with_relation_only_all_losses_custom.py',
    'swin_tiny': 'projects/configs/co_dino_swin/co_dino_5scale_swin_tiny_coco_with_relation_only_all_losses_custom.py',
    'swin_small': 'projects/configs/co_dino_swin/co_dino_5scale_swin_small_coco_with_relation_only_all_losses_custom.py',
    'resnet50': 'projects/configs/co_dino_swin/co_dino_5scale_r50_coco_with_relation_only_all_losses_custom.py',
}

TRAIN_ANNOTATION = 'annotations/train_h_first_second_full_corrected_auto_80_100.json'
VAL_ANNOTATION_CANDIDATES = [
    'annotations/val_h_first_second_full_corrected_w_area.json',
    'annotations/val_h_first_second_full_10.json',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train HOI-DETR with a selectable backbone config')
    parser.add_argument(
        '--backbone',
        choices=sorted(BACKBONE_CONFIGS),
        default='swin_tiny',
        help='Backbone profile to train with')
    parser.add_argument(
        '--config',
        help='Explicit config path. Overrides --backbone when provided.')
    parser.add_argument(
        '--pretrained',
        help='Optional backbone checkpoint path. Passed via --cfg-options.')
    parser.add_argument(
        '--no-pretrained',
        action='store_true',
        help='Force model.backbone.pretrained=None via --cfg-options.')
    parser.add_argument(
        '--data-root',
        help='Override data_root in config via --cfg-options.')
    return parser.parse_known_args()


def check_mmlab_stack():
    if importlib_util.find_spec('mmcv') is None:
        raise SystemExit(
            'MMCV is not installed in this environment. This repo expects '
            'MMDetection v2.25.3 with mmcv-full==1.5.0. Install with: '
            'python -m pip install "setuptools<82" wheel && '
            'python -m pip install mmcv-full==1.5.0 '
            '-f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html')

    try:
        numpy_version = version('numpy')
        numpy_major = int(numpy_version.split('.', 1)[0])
    except (PackageNotFoundError, ValueError):
        numpy_version = None
        numpy_major = None

    if numpy_major is not None and numpy_major >= 2:
        raise SystemExit(
            f'Incompatible NumPy version detected ({numpy_version}). '
            'This stack expects NumPy 1.x. Run: '
            'python -m pip install "numpy<2"')


def resolve_config(repo_root, args):
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = repo_root / config_path
        return config_path

    relative_config = BACKBONE_CONFIGS[args.backbone]
    return repo_root / relative_config


def read_default_data_root(config_path):
    pattern = re.compile(r"^data_root\s*=\s*['\"]([^'\"]+)['\"]", re.MULTILINE)
    text = config_path.read_text(encoding='utf-8')
    match = pattern.search(text)
    if not match:
        return None
    return match.group(1)


def validate_data_root(data_root):
    root = Path(data_root)
    if not root.exists():
        raise SystemExit(
            f'Dataset root not found: {root}. '
            'Download Hands23 locally and pass --data-root to this script.')

    missing = []
    if not (root / TRAIN_ANNOTATION).exists():
        missing.append(TRAIN_ANNOTATION)

    has_val = any((root / rel).exists() for rel in VAL_ANNOTATION_CANDIDATES)
    if not has_val:
        missing.append('one of: ' + ', '.join(VAL_ANNOTATION_CANDIDATES))

    if missing:
        pretty_missing = '; '.join(missing)
        raise SystemExit(
            f'Dataset root exists but required annotation files are missing under {root}: '
            f'{pretty_missing}.')


def main():
    args, passthrough_args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_path = resolve_config(repo_root, args)

    check_mmlab_stack()

    if not config_path.exists():
        available = ', '.join(sorted(BACKBONE_CONFIGS))
        raise FileNotFoundError(
            f'Config not found: {config_path}. ' \
            f'Add the matching config file or pass --config explicitly. ' \
            f'Available backbone presets: {available}.')

    cfg_overrides = []
    if args.no_pretrained:
        cfg_overrides.append('model.backbone.pretrained=None')
    elif args.pretrained:
        cfg_overrides.append(f'model.backbone.pretrained={args.pretrained}')

    if args.data_root:
        data_root = args.data_root
        if not data_root.endswith('/'):
            data_root += '/'
        validate_data_root(data_root)
        cfg_overrides.append(f'data_root={data_root}')
    else:
        default_data_root = read_default_data_root(config_path)
        if default_data_root and not Path(default_data_root).exists():
            raise SystemExit(
                'Config default data_root does not exist on this machine: '
                f'{default_data_root}. Pass --data-root with your local Hands23 path.')

    if cfg_overrides:
        passthrough_args = [*passthrough_args, '--cfg-options', *cfg_overrides]

    train_script = repo_root / 'tools' / 'train.py'
    command = [sys.executable, str(train_script), str(config_path), *passthrough_args]
    raise SystemExit(subprocess.run(command).returncode)


if __name__ == '__main__':
    main()