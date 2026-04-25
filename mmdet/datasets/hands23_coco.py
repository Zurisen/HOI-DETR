from mmdet.datasets import CocoDataset
import numpy as np
from .builder import DATASETS

@DATASETS.register_module()
class Hands23CocoDataset(CocoDataset):
    """COCO dataset with extra handside and interaction fields."""

    def _parse_ann_info(self, img_info, ann_info):
        """Parse annotations and include handside/interaction fields."""
        gt_bboxes = []
        gt_labels = []
        gt_handside = []
        gt_interaction = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for ann in ann_info:
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['iscrowd']:
                gt_bboxes_ignore.append([x1, y1, x1 + w, y1 + h])
            else:
                gt_bboxes.append([x1, y1, x1 + w, y1 + h])
                gt_labels.append(self.cat2label[ann['category_id']])
                
                
                interaction_val = ann.get('interaction', (-1,-1)) # Get value, default to -1
    
                if interaction_val == -1:
                    # If no interaction, use the negative of the object's own ID
                    final_interaction_val = (ann['id'], -1)
                else:
                    # Otherwise, use the positive interaction ID
                    final_interaction_val = (ann['id'],interaction_val)
                    
                    
                gt_handside.append(ann.get('handside', -1))  # <--- Custom field      
                gt_interaction.append(final_interaction_val)  
                #print('ann.get(interaction, -1): ',final_interaction_val)        
                #import pdb
                #pdb.set_trace()
                #gt_interaction.append(ann.get('interaction', -1))  # <--- Custom field
                gt_masks_ann.append(ann.get('segmentation', []))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_handside = np.array(gt_handside, dtype=np.int64)
            gt_interaction = np.array(gt_interaction, dtype=np.int64)
            
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_handside = np.array([], dtype=np.int64)
            gt_interaction = np.array([], dtype=np.int64)
            

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        assert gt_bboxes_ignore.shape[0] == 0
        assert gt_bboxes.shape[0] == gt_labels.shape[0] == gt_handside.shape[0]
        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=img_info.get('segm_file', None),  # Use .get() to avoid KeyError
            handside=gt_handside,  # <--- Add to ann dict
            interaction=gt_interaction
        )
        return ann

    def prepare_train_img(self, idx):
        results = super().prepare_train_img(idx)
        ann = self.get_ann_info(idx)
        #results['gt_handside'] = ann.get('handside', np.array([], dtype=np.int64))
        #print(f"[DEBUG] prepare_train_img: gt_handside = {results['gt_handside']}")
        return results

    def prepare_test_img(self, idx):
        """Prepare test data (no GT annotations)."""
        return super().prepare_test_img(idx)
