# mmdet/datasets/collate_fn.py
import collections
from mmcv.parallel import DataContainer as DC
from torch.utils.data.dataloader import default_collate
import torch
import numpy as np
import logging

# Get MMDetection's logger
logger = logging.getLogger('mmdet')

# Add this print to confirm it's being used
logger.info("!!! Using Custom Collate Function vX.X !!!") # Update version number for tracking

def custom_collate(batch):
    """
    Custom collate function for MMDetection 2.x to handle batching of
    variable-length ground truth data (bboxes, labels, custom fields)
    and ensuring correct image stacking.
    """
    logger.debug(f"--- custom_collate called for batch of size {len(batch)} ---")

    if not isinstance(batch, collections.abc.Sequence):
        logger.error(f"Batch is not a sequence: {type(batch)}")
        return default_collate(batch) # Fallback, but likely error

    batched_data = {}
    
    # Define keys that should always remain lists of tensors (per image in batch)
    # These are typically variable-length instance-level data
    variable_length_gt_keys = [
        'gt_bboxes', 'gt_labels', 'gt_handside', 'gt_bboxes_ignore', 'proposals',
        'gt_masks','gt_interaction', # Masks are also typically lists of per-instance data
    ]

    # Iterate over all possible keys that could be present in a processed sample
    # from the pipeline.
    # We collect all unique keys from all samples in the batch to be robust.
    all_keys_in_batch_samples = set()
    for s in batch:
        all_keys_in_batch_samples.update(s.keys())
    
    for key in sorted(list(all_keys_in_batch_samples)): # Sort for consistent debug output
        # Skip keys that are handled by specific logic below or shouldn't be touched by loop
        if key in ['img', 'img_metas', 'gt_semantic_seg']: # Handle these explicitly
            continue 

        # Retrieve the DataContainer item for this key from all samples in the batch
        items_for_this_key = []
        for s_idx, s in enumerate(batch):
            if key not in s:
                # If a key is missing from a sample, it's problematic for batching.
                # This often indicates an issue in LoadAnnotations/DefaultFormatBundle.
                # For safety, let's append None, and convert it later.
                items_for_this_key.append(None)
                logger.warning(f"Key '{key}' missing from sample {s_idx} in batch.")
                continue

            dc_item = s[key]
            # Extract the raw data from DataContainer
            data_item = dc_item.data if isinstance(dc_item, DC) else dc_item
            items_for_this_key.append(data_item)

        # --- Process based on the key type and expected collation ---

        if key in variable_length_gt_keys:
            # These are lists of tensors (one tensor per image), should NOT be stacked by default_collate.
            processed_items = []
            for item in items_for_this_key:
                if item is None:
                    # Provide appropriate empty tensor based on the key
                    if key == 'gt_labels' or key == 'gt_handside':
                        processed_items.append(torch.empty((0,), dtype=torch.long))
                    elif key == 'gt_bboxes' or key == 'gt_bboxes_ignore' or key == 'proposals':
                        processed_items.append(torch.empty((0, 4), dtype=torch.float32))
                    elif key == 'gt_masks': # Masks are typically lists of np.arrays or list of list of polys
                        processed_items.append([]) # Empty list of masks
                    else:
                        processed_items.append(item) # Keep None if type is unknown for safety
                elif isinstance(item, np.ndarray):
                    # Convert numpy arrays to tensors with appropriate dtype
                    if 'label' in key or 'handside' in key:
                        processed_items.append(torch.from_numpy(item).to(torch.long))
                    elif 'bbox' in key or 'proposal' in key:
                        processed_items.append(torch.from_numpy(item).to(torch.float32))
                    else:
                        processed_items.append(torch.from_numpy(item))
                elif torch.is_tensor(item):
                    processed_items.append(item)
                else: # For other types, e.g., list of lists for masks polygons
                    processed_items.append(item)
            
            # CRITICAL: Wrap the list of already-processed tensors/lists directly in DataContainer.
            # DO NOT call default_collate on this list, as items have variable sizes.
            batched_data[key] = DC(processed_items, stack=False)
            logger.debug(f"  Key '{key}' handled as variable-length GTs. Sample 0 type: {type(processed_items[0]) if processed_items else 'Empty'}. Shape: {processed_items[0].shape if processed_items and torch.is_tensor(processed_items[0]) else 'N/A'}")

        # This `else` block is *gone*. Every key must be handled explicitly.
        # This prevents accidental `default_collate` calls on variable-length data.

    # --- Explicitly handle common top-level keys that need specific collation ---

    # `img_metas` (List[dict] -> DC(List[dict]))
    if 'img_metas' in all_keys_in_batch_samples:
        batched_data['img_metas'] = DC([s['img_metas'].data for s in batch], cpu_only=True)
        logger.debug(f"  Key 'img_metas' handled as cpu_only list.")
    
    # `img` (List[Tensor] -> DC(Tensor)) - these MUST be stackable (same H,W)
    if 'img' in all_keys_in_batch_samples:
        try:
            batched_data['img'] = DC(default_collate([s['img'].data for s in batch]), stack=True)
            logger.debug(f"  Key 'img' successfully stacked. Shape: {batched_data['img'].data.shape}")
        except RuntimeError as e:
            logger.error(f"CRITICAL ERROR: 'img' failed to stack in custom_collate: {e}")
            for s_idx, s in enumerate(batch):
                img_data = s['img'].data if isinstance(s['img'], DC) else s['img']
                logger.error(f"  Sample {s_idx} img shape: {img_data.shape}, dtype: {img_data.dtype}")
            raise e
            
    # `gt_semantic_seg` (often (1, H, W) after formatting, should be stackable if H,W are consistent)
    if 'gt_semantic_seg' in all_keys_in_batch_samples:
        try:
            # Need to get raw tensor data from DC for default_collate
            sem_seg_items = []
            for s in batch:
                if 'gt_semantic_seg' in s and isinstance(s['gt_semantic_seg'], DC):
                    sem_seg_items.append(s['gt_semantic_seg'].data)
                elif 'gt_semantic_seg' in s and isinstance(s['gt_semantic_seg'], np.ndarray): # Handle numpy
                    sem_seg_items.append(torch.from_numpy(s['gt_semantic_seg']))
                else: # Fallback for missing/unexpected
                    sem_seg_items.append(None) # Or empty tensor
            
            # Filter Nones before collate, or ensure default_collate handles them.
            # Assuming DefaultFormatBundle makes them (1,H,W) or empty.
            valid_sem_seg_items = [item for item in sem_seg_items if item is not None and item.numel() > 0]
            if len(valid_sem_seg_items) == len(sem_seg_items): # All valid, no missing
                batched_data['gt_semantic_seg'] = DC(default_collate(sem_seg_items), stack=True)
                logger.debug(f"  Key 'gt_semantic_seg' successfully stacked. Shape: {batched_data['gt_semantic_seg'].data.shape}")
            else: # Mixed/missing, treat as variable length
                processed_sem_seg = []
                for item in sem_seg_items:
                    if item is None or (torch.is_tensor(item) and item.numel() == 0):
                        processed_sem_seg.append(torch.empty((0,0,0), dtype=torch.float32)) # Example empty
                    else:
                        processed_sem_seg.append(item)
                batched_data['gt_semantic_seg'] = DC(processed_sem_seg, stack=False)
                logger.debug(f"  Key 'gt_semantic_seg' handled as variable-length.")

        except RuntimeError as e:
            logger.error(f"CRITICAL ERROR: 'gt_semantic_seg' failed to stack: {e}")
            for s_idx, s in enumerate(batch):
                sem_seg_data = s['gt_semantic_seg'].data if isinstance(s['gt_semantic_seg'], DC) else s['gt_semantic_seg']
                logger.error(f"  Sample {s_idx} gt_semantic_seg shape: {sem_seg_data.shape if torch.is_tensor(sem_seg_data) else 'N/A'}")
            raise e


    # --- Final Check for any unhandled keys that might cause issues ---
    # This loop is mostly for debugging, to catch anything missed.
    for key in all_keys_in_batch_samples:
        if key not in batched_data:
            logger.warning(f"Key '{key}' was present in input batch but not explicitly handled by custom_collate. It might be dropped or cause issues.")
    
    logger.debug("--- custom_collate finished ---")
    return batched_data