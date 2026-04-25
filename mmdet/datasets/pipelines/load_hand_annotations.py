from mmdet.datasets.pipelines import LoadAnnotations
from ..builder import PIPELINES
import numpy as np

@PIPELINES.register_module()
class LoadHandAnnotations(LoadAnnotations):
    """Custom LoadAnnotations to also handle handside without filtering."""

    def _load_handside(self, results):
        ann_info = results['ann_info']

        # Store the raw handsides and raw boxes for later matching
        results['gt_handside'] = np.array(ann_info.get('handside', []), dtype=np.int32)
        results['gt_bboxes_all'] = np.array(ann_info.get('bboxes', []), dtype=np.float32)
        results['gt_interaction'] = np.array(ann_info.get('interaction', []), dtype=np.int32)
        return results

    def __call__(self, results):
        results = super().__call__(results)  # Loads and filters gt_bboxes, gt_labels
        results = self._load_handside(results)
        return results
