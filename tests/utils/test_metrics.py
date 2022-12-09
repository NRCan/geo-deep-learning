from typing import List, Dict, Tuple

import numpy as np

from utils.metrics import AverageMeter
from utils.metrics import create_metrics_dict, calculate_batch_metrics, calculate_confusion_matrix
import torch
import pytest
from _pytest.fixtures import SubRequest


class TestAverageMeter:
    @pytest.fixture(scope="class", params=zip([[1, 2, 3, 4], [1, 1, 1, 1], [0, 0, 0, 0]], [3, 2, 1]))
    def samples(self, request: SubRequest) -> List:
        return [request.param[0], request.param[1]]

    def test_update(self, samples: List) -> None:
        average_meter = AverageMeter()
        for value in samples[0]:
            average_meter.update(
                val=value,
                batch_size=samples[1]
            )
        assert average_meter.total_batches == len(samples[0])
        assert average_meter.total_samples == len(samples[0]) * samples[1]
        assert average_meter.total_metrics_sum == sum(samples[0])
        assert average_meter.avg == sum(samples[0]) / len(samples[0])

    def test_average(self, samples: List) -> None:
        average_meter = AverageMeter()

        assert average_meter.average() == 0.0

        for value in samples[0]:
            average_meter.update(
                val=value,
                batch_size=samples[1]
            )
        assert average_meter.average() == sum(samples[0]) / len(samples[0])

    def test_reset(self, samples: List) -> None:
        average_meter = AverageMeter()
        for value in samples[0]:
            average_meter.update(
                val=value,
                batch_size=samples[1]
            )

        average_meter.reset()
        assert average_meter.total_batches == 0.0
        assert average_meter.total_samples == 0.0
        assert average_meter.total_metrics_sum == 0.0
        assert average_meter.avg == 0.0


class TestMetrics:
    @staticmethod
    def initialize_validation_dict(num_classes: int = 1) -> Dict:
        num_classes = 2 if num_classes == 1 else num_classes
        val_dict = {"iou": 0.0, "precision": 0.0, "recall": 0.0, "fscore": 0.0, "iou_nonbg": 0.0}

        for key in val_dict.copy().keys():
            if key == "iou_nonbg":
                continue
            per_class_vals = {key + "_" + str(i): 0.0 for i in range(num_classes)}
            val_dict.update(per_class_vals)

        return val_dict

    @pytest.fixture(params=[0, 1, 2, 100])
    def num_classes(self, request: SubRequest) -> int:
        return request.param

    def test_create_metrics_dict(self, num_classes: int) -> None:
        """Evaluate the metrics dictionary creation.
        Binary and multiclass"""
        skip_keys = ['loss']
        if num_classes < 1:
            with pytest.raises(AssertionError):
                metrics_dict = create_metrics_dict(num_classes)
        else:
            metrics_dict = create_metrics_dict(num_classes)
            val_dict = self.initialize_validation_dict(num_classes)

            for key, value in metrics_dict.items():
                if key in skip_keys:
                    continue
                assert key in val_dict
                assert isinstance(value, AverageMeter)

    @pytest.fixture(scope="class")
    def binary_samples(self) -> List:
        logits = []
        labels = []
        vals = []

        logit = torch.reshape(torch.tensor([[[-2, -1, 0], [0, 1, 2], [2, -0.5, -2]],
                                            [[-2, 0, -2], [1, 0.5, 1], [1, -0.5, 1]]]), shape=(2, 1, 3, 3))
        label = torch.reshape(torch.tensor([[[1, 0, 1], [1, 1, 1], [1, 0, 0]],
                                            [[0, 0, 1], [1, 0, 1], [1, 0, 1]]]), shape=(2, 3, 3))
        validation = {
            "iou": 0.5643939393939394,
            "precision": 0.7467532467532467,
            "recall": 0.7375,
            "fscore": 0.7213622291021671,
            "iou_nonbg": 0.5833333333333334,
            "iou_0": 0.5454545454545454,
            "iou_1": 0.5833333333333334,
            "precision_0": 0.8571428571428571,
            "precision_1": 0.6363636363636364,
            "recall_0": 0.6,
            "recall_1": 0.875,
            "fscore_0": 0.7058823529411764,
            "fscore_1": 0.7368421052631579
        }

        logits.append(logit)
        labels.append(label)
        vals.append(validation)

        # 2nd set of test samples with ignor index values (-1, 255):
        logit = torch.reshape(torch.tensor([[[0, -2.5, -1.5], [0.5, 3, 1], [1, -0.5, -2]],
                                            [[-2, 0.5, 2], [-1.5, 1.5, 0], [1, -1.5, 1]]]), shape=(2, 1, 3, 3))
        label = torch.reshape(torch.tensor([[[255, 0, 0], [1, 1, 255], [1, 1, 0]],
                                            [[0, 1, 1], [1, 0, 1], [1, 0, -1]]]), shape=(2, 3, 3))
        validation = {
            "iou": 0.5777777777777777,
            "precision": 0.75,
            "recall": 0.7410714285714286,
            "fscore": 0.7321428571428572,
            "iou_nonbg": 0.6,
            "iou_0": 0.5555555555555556,
            "iou_1": 0.6,
            "precision_0": 0.8333333333333334,
            "precision_1": 0.6666666666666666,
            "recall_0": 0.625,
            "recall_1": 0.8571428571428571,
            "fscore_0": 0.7142857142857143,
            "fscore_1": 0.75
        }

        logits.append(logit)
        labels.append(label)
        vals.append(validation)

        return [logits, labels, vals]

    def test_calculate_batch_metrics_binary(self, binary_samples: List) -> None:
        metrics_dict = create_metrics_dict(1)
        val_values_dict = self.initialize_validation_dict()

        i = 0
        for logit, label, value in zip(*binary_samples):
            metrics_dict = calculate_batch_metrics(
                predictions=logit,
                gts=label,
                n_classes=1,
                metric_dict=metrics_dict
            )
            for key in val_values_dict.keys():
                val_values_dict[key] += value[key]
            i += 1

        for key in val_values_dict.keys():
            val_values_dict[key] /= i

        for key in metrics_dict.keys():
            if key == 'loss':
                continue
            assert "{:.6f}".format(metrics_dict[key].average()) == "{:.6f}".format(val_values_dict[key])

    @pytest.fixture(scope="class")
    def multi_samples(self) -> List:
        logits = []
        labels = []
        vals = []

        logit = torch.reshape(torch.tensor([[[-2, -1, 0], [0, 1.5, 0], [2, -0.5, 2]],
                                            [[-1, 0, -2], [1, .5, 1], [1, -0.5, 1]],
                                            [[2, 1, 2], [.5, 1, 0.25], [0, -0.5, -2]]]), shape=(1, 3, 3, 3))
        label = torch.reshape(torch.tensor([[2, 2, 2], [1, 1, 1], [0, 0, 0]]), shape=(1, 3, 3))
        validation = {
            "iou": 0.8055555555555555,
            "precision": 0.8888888888888888,
            "recall": 0.9166666666666666,
            "fscore": 0.8857142857142857,
            "iou_nonbg": 0.8333333333333333,
            "iou_0": 0.75,
            "iou_1": 0.6666666666666666,
            "iou_2": 1.0,
            "precision_0": 1.0,
            "precision_1": 0.6666666666666666,
            "precision_2": 1.0,
            "recall_0": 0.75,
            "recall_1": 1.0,
            "recall_2": 1.0,
            "fscore_0": 0.8571428571428571,
            "fscore_1": 0.8,
            "fscore_2": 1.0
        }

        logits.append(logit)
        labels.append(label)
        vals.append(validation)

        # 2nd set of test samples with ignor index values (-1, 255):
        logit = torch.reshape(torch.tensor([[[2, -1, 0], [0, 1.5, 0], [2, 0.5, 2]],
                                           [[-1, 0, -2], [1, 1.5, 1], [1, 2.5, 1]],
                                           [[1, 1, 2], [.5, 1, 0.25], [0, -0.5, -2]]]), shape=(1, 3, 3, 3))
        label = torch.reshape(torch.tensor([[255, 2, 2], [1, -1, 1], [0, 0, 255]]), shape=(1, 3, 3))
        validation = {
            "iou": 0.7222222222222222,
            "precision": 0.8333333333333334,
            "recall": 0.8888888888888888,
            "fscore": 0.8222222222222223,
            "iou_nonbg": 0.8333333333333333,
            "iou_0": 0.5,
            "iou_1": 0.6666666666666666,
            "iou_2": 1.0,
            "precision_0": 0.5,
            "precision_1": 1.0,
            "precision_2": 1.0,
            "recall_0": 1.0,
            "recall_1": 0.6666666666666666,
            "recall_2": 1.0,
            "fscore_0": 0.6666666666666666,
            "fscore_1": 0.8,
            "fscore_2": 1.0,

        }

        logits.append(logit)
        labels.append(label)
        vals.append(validation)

        return [logits, labels, vals]

    def test_calculate_batch_metrics_multi(self, multi_samples: List) -> None:
        metrics_dict = create_metrics_dict(num_classes=3)
        val_values_dict = self.initialize_validation_dict(num_classes=3)

        i = 0
        for logit, label, value in zip(*multi_samples):
            metrics_dict = calculate_batch_metrics(
                predictions=logit,
                gts=label,
                n_classes=3,
                metric_dict=metrics_dict
            )
            for key in val_values_dict.keys():
                val_values_dict[key] += value[key]
            i += 1

        for key in val_values_dict.keys():
            val_values_dict[key] /= i

        for key in metrics_dict.keys():
            if key == 'loss':
                continue
            assert "{:.6f}".format(metrics_dict[key].average()) == "{:.6f}".format(val_values_dict[key])

    @pytest.fixture(params=zip([[0, 1, 0, 1, 1], [1, 1, 1, 0, 1], [0, 1, 2, 3, 3]],
                               [[0, 1, 1, 1, 0], [-1, 1, 1, 0, 1], [0, 2, 1, 3, 255]],
                               [1, 1, 4],
                               [[[1, 1], [1, 2]], [[1, 0], [0, 3]], [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]]))
    def cm_inputs(self, request: SubRequest) -> Tuple:
        return request.param

    def test_calculate_confusion_matrix(self, cm_inputs: List) -> None:
        predicted = np.array(cm_inputs[0])
        ground_truth = np.array(cm_inputs[1])
        n_classes = 2 if cm_inputs[2] == 1 else cm_inputs[2]
        assert calculate_confusion_matrix(
            label_pred=predicted,
            label_true=ground_truth,
            n_classes=n_classes).all() == np.array(cm_inputs[3]).all()



