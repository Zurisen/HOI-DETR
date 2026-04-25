"""
demo.py
-------
Co-DETR hand-object interaction demo.

Edit the variables at the top for your paths, then run:
    python demo/demo.py
"""

import glob
import os

import mmcv
from tqdm import tqdm

from mmdet.apis               import init_detector
from mmdet.datasets.pipelines import Compose

from projects import *  # noqa: F401,F403  (registers Co-DETR custom modules)

from configs import CLASS_NAMES
from helpers import (
    find_interaction_branch,
    run_inference,
    call_interaction,
    compute_style,
    draw_ui,
)


# ══════════════════════════════════════════════════════════════
# USER SETTINGS
# ══════════════════════════════════════════════════════════════
MODEL_CONFIG = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco_with_relation_only_all_losses_custom.py'
CHECKPOINT   = 'checkpoints/epoch_5.pth'
DEVICE       = 'cuda:0'

# Input: a directory of images (any folder name; final path segment is
# reused in the default output directory).
INPUT_DIR    = 'demo/example_images2'


# Output: None  -> demo/results/<basename(INPUT_DIR)>/  (recommended)
#         str   -> use that exact directory
OUTPUT_DIR   = None

# Detection thresholds
SCORE_THR    = 0.3
NMS_IOU      = 0.5

# Visualisation mode
# VERBOSE_LABELS = False -> smart hiding: labels/badges on tiny boxes and
#                           short links are suppressed (cleaner output).
# VERBOSE_LABELS = True  -> show every label and every probability badge,
#                           regardless of size (use for debugging or when
#                           you want the raw, complete view).
VERBOSE_LABELS = True


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    # Resolve output directory
    out_dir = OUTPUT_DIR or os.path.join(
        'demo', 'results', os.path.basename(os.path.normpath(INPUT_DIR))
    )
    os.makedirs(out_dir, exist_ok=True)

    # Collect input images (preserve filenames on output)
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_list = sorted(
        f for ext in exts for f in glob.glob(os.path.join(INPUT_DIR, ext))
    )
    if not image_list:
        print(f"[ERROR] No images found in {INPUT_DIR}")
        return
    print(f"[INFO] {len(image_list)} image(s) from {INPUT_DIR}")
    print(f"[INFO] Saving to {out_dir}")

    # Build model and pipeline
    model = init_detector(MODEL_CONFIG, CHECKPOINT, device=DEVICE)
    model.eval()
    test_pipeline      = Compose(model.cfg.data.test.pipeline)
    interaction_branch = find_interaction_branch(model.query_head)
    print(f"[INFO] Interaction MLP input dim: "
          f"{interaction_branch.mlp[0].in_features}")

    # Main loop
    for img_path in tqdm(sorted(image_list)):
        orig_img = mmcv.imread(img_path)

        try:
            dets, embeds = run_inference(
                model, test_pipeline, img_path,
                device      = DEVICE,
                class_names = CLASS_NAMES,
                score_thr   = SCORE_THR,
                nms_iou     = NMS_IOU,
            )
        except Exception as e:
            print(f"[ERROR] {img_path}: {e}")
            continue

        if not dets:
            continue

        # Predict interactions: all H->F and F->S pairs
        hands   = [d for d in dets if d['class_id'] == 0]
        firsts  = [d for d in dets if d['class_id'] == 1]
        seconds = [d for d in dets if d['class_id'] == 2]

        hf_inters, fs_inters = [], []
        for h in hands:
            for f in firsts:
                interacts, prob = call_interaction(
                    interaction_branch,
                    embeds[h['query_idx']], embeds[f['query_idx']],
                )
                if interacts:
                    hf_inters.append((h, f, prob))
        for f in firsts:
            for so in seconds:
                interacts, prob = call_interaction(
                    interaction_branch,
                    embeds[f['query_idx']], embeds[so['query_idx']],
                )
                if interacts:
                    fs_inters.append((f, so, prob))

        # Render (in-place on `vis`) and save with the original filename
        vis   = orig_img.copy()
        style = compute_style(vis.shape)
        draw_ui(vis, dets, hf_inters, fs_inters, style,
                verbose_labels=VERBOSE_LABELS)

        out_path = os.path.join(out_dir, os.path.basename(img_path))
        mmcv.imwrite(vis, out_path)

    print(f"[INFO] Done. Results saved to {out_dir}")


if __name__ == '__main__':
    main()
