from utils.metrics import create_metrics_dict, report_classification, iou
import torch
import pytest

# Test arrays: [bs=2, h=2, w,2]
def init_tensors():
    pred_multi = torch.tensor([0, 0, 2, 2, 0, 2, 1, 2, 1, 0, 2, 2, 1, 0, 2, 2])
    pred_binary = torch.tensor([0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1])

    lbl_multi = torch.tensor([1, 0, 2, 2, 0, 1, 2, 0, 2, 2, 0, 0, 1, 2, 0, 1])
    lbl_binary = torch.tensor([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    # array with dont care
    lbl_multi_dc = torch.tensor([-1, -1, 2, 2, 0, 1, 2, 0, 2, 2, 0, 0, 1, 2, 0, 1])
    lbl_binary_dc = torch.tensor([-1, -1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1])
    return {'pred_multi': pred_multi, 
            'pred_binary': pred_binary, 
            'lbl_multi': lbl_multi, 
            'lbl_binary': lbl_binary, 
            'lbl_multi_dc': lbl_multi_dc, 
            'lbl_binary_dc': lbl_binary_dc}


class TestMetrics(object):
    def test_create_metrics_dict(self):
        """Evaluate the metrics dictionnary creation. 
        Binary and multiclass"""
        # binary tasks have 1 class at class definition.
        num_classes = 1
        metrics_dict = create_metrics_dict(num_classes)
        assert 'iou_1' in metrics_dict.keys()
        assert 'iou_2' not in metrics_dict.keys()

        num_classes = 3
        metrics_dict = create_metrics_dict(num_classes)
        assert 'iou_1' in metrics_dict.keys()
        assert 'iou_2' in metrics_dict.keys()
        assert 'iou_3' not in metrics_dict.keys()
        del metrics_dict

    def test_report_classification_multi(self):
        """Evaluate report classification. 
        Multiclass, without ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        metrics_dict = report_classification(t['pred_multi'], 
                                             t['lbl_multi'], 
                                             batch_size=2, 
                                             metrics_dict=metrics_dict,
                                             ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['precision'].val) == "0.327083"
        assert "{:.6f}".format(metrics_dict['recall'].val) == "0.312500"
        assert "{:.6f}".format(metrics_dict['fscore'].val) == "0.314935"

    def test_report_classification_multi_ignore_idx(self):
        """Evaluate report classification. 
        Multiclass, with ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        metrics_dict = report_classification(t['pred_multi'], 
                                             t['lbl_multi_dc'], 
                                             batch_size=2, 
                                             metrics_dict=metrics_dict,
                                             ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['precision'].val) == "0.297619"
        assert "{:.6f}".format(metrics_dict['recall'].val) == "0.285714"
        assert "{:.6f}".format(metrics_dict['fscore'].val) == "0.283163"

    def test_report_classification_binary(self):
        """Evaluate report classification. 
        Binary, without ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(1)
        metrics_dict = report_classification(t['pred_binary'], 
                                             t['lbl_binary'], 
                                             batch_size=2, 
                                             metrics_dict=metrics_dict,
                                             ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['precision'].val) == "0.547727"
        assert "{:.6f}".format(metrics_dict['recall'].val) == "0.562500"
        assert "{:.6f}".format(metrics_dict['fscore'].val) == "0.553030"

    def test_report_classification_binary_ignore_idx(self):
        """Evaluate report classification. 
        Binary, without ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(1)
        metrics_dict = report_classification(t['pred_binary'], 
                                             t['lbl_binary_dc'], 
                                             batch_size=2, 
                                             metrics_dict=metrics_dict,
                                             ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['precision'].val) == "0.528139"
        assert "{:.6f}".format(metrics_dict['recall'].val) == "0.571429"
        assert "{:.6f}".format(metrics_dict['fscore'].val) == "0.539286"

    def test_iou_multi(self):
        """Evaluate iou calculation. 
        Multiclass, without ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        metrics_dict = iou(t['pred_multi'], 
                           t['lbl_multi'], 
                           batch_size=2, 
                           num_classes=3,
                           metric_dict=metrics_dict,
                           ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['iou'].val) == "0.185185"
    
    def test_iou_multi_ignore_idx(self):
        """Evaluate iou calculation. 
        Multiclass, with ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        # wih ignore_index == -1
        metrics_dict = iou(t['pred_multi'], 
                           t['lbl_multi_dc'], 
                           batch_size=2, 
                           num_classes=3,
                           metric_dict=metrics_dict,
                           ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['iou'].val) == "0.233333"

        # with ignore_index == 0
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        metrics_dict = iou(t['pred_multi'], 
                           t['lbl_multi'], 
                           batch_size=2, 
                           num_classes=3,
                           metric_dict=metrics_dict,
                           ignore_index=0)
        assert "{:.6f}".format(metrics_dict['iou'].val) == "0.208333"

    def test_iou_binary(self):
        """Evaluate iou calculation. 
        Binary, without ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(1)
        metrics_dict = iou(t['pred_binary'], 
                           t['lbl_binary'], 
                           batch_size=2, 
                           num_classes=1,
                           metric_dict=metrics_dict,
                           ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['iou'].val) == "0.361111" 

    def test_iou_binary_ignore_idx(self):
        """Evaluate iou calculation. 
        Binary, with ignore_index in array."""
        t = init_tensors()
        metrics_dict = create_metrics_dict(1)
        # with ignore_index == -1
        metrics_dict = iou(t['pred_binary'], 
                           t['lbl_binary_dc'], 
                           batch_size=2, 
                           num_classes=1,
                           metric_dict=metrics_dict,
                           ignore_index=-1)
        assert "{:.6f}".format(metrics_dict['iou'].val) == "0.435897"

        # with ignore_index == 0
        t = init_tensors()
        metrics_dict = create_metrics_dict(3)
        with pytest.raises(ValueError):
            metrics_dict =  iou(t['pred_binary'], 
                                t['lbl_binary_dc'], 
                                batch_size=2, 
                                num_classes=3,
                                metric_dict=metrics_dict,
                                ignore_index=0)