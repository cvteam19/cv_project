"""Final agreed YOLO: Faster R-CNN input transform, SGD, three-hour budget.

Pinned to ultralytics 8.3.221. Embedded into the standalone Kaggle notebook.
The runtime checks must pass on Kaggle before training.
"""
from copy import copy
from pathlib import Path
import random
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from ultralytics.data import build_dataloader
from ultralytics.models.yolo.detect import DetectionTrainer, DetectionValidator
from ultralytics.utils import ops
from budget_protocol import budget_cosine_lr, set_budget_lr

BASELINE_MEAN = [0.485, 0.456, 0.406]
BASELINE_STD = [0.229, 0.224, 0.225]


def input_transform():
    # Same normalization, interpolation, resize rounding, zero padding and stride
    # as Faster R-CNN's native GeneralizedRCNNTransform.
    return GeneralizedRCNNTransform(
        min_size=640, max_size=640, image_mean=BASELINE_MEAN,
        image_std=BASELINE_STD, size_divisible=32)


def flip_image_target(image, boxes):
    reflected = boxes.clone()
    reflected[:, 0] = image.shape[-1] - boxes[:, 2]
    reflected[:, 2] = image.shape[-1] - boxes[:, 0]
    return image.flip(-1), reflected


def collate_baseline_input(samples):
    tensors, targets, filenames, shapes = zip(*samples)
    transform = input_transform()
    # min_size has one value, so train/eval select exactly the same size.
    transformed, targets = transform(list(tensors), list(targets))
    height, width = transformed.tensors.shape[-2:]
    boxes, classes, indices = [], [], []
    for index, target in enumerate(targets):
        xywh = ops.xyxy2xywh(target['boxes'])
        xywh /= xywh.new_tensor([width, height, width, height])
        boxes.append(xywh)
        classes.append(target['labels'].float().reshape(-1, 1))
        indices.append(torch.full((len(xywh),), index, dtype=torch.float32))
    return dict(
        img=transformed.tensors, bboxes=torch.cat(boxes),
        cls=torch.cat(classes), batch_idx=torch.cat(indices),
        im_file=filenames, ori_shape=shapes,
        resized_shape=transformed.image_sizes,
        ratio_pad=tuple(((h1 / h0, w1 / w0), (0, 0)) for
                        (h0, w0), (h1, w1) in zip(shapes, transformed.image_sizes)),
    )


class BaselineInputDataset(Dataset):
    rect = False
    collate_fn = staticmethod(collate_baseline_input)

    def __init__(self, image_dir, train=False):
        self.train = train
        image_dir = Path(image_dir)
        assert image_dir.is_dir(), f'Expected an images directory: {image_dir}'
        self.im_files = sorted(str(p) for p in image_dir.rglob('*')
                               if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'})
        self.labels = []
        for filename in self.im_files:
            relative = Path(filename).relative_to(image_dir).with_suffix('.txt')
            label = image_dir.parent / 'labels' / relative
            rows = np.array([list(map(float, line.split())) for line in
                             label.read_text().splitlines() if line.strip()], dtype=np.float32).reshape(-1, 5)
            self.labels.append(dict(cls=rows[:, :1], bboxes=rows[:, 1:]))
        assert self.im_files, 'No images found'

    def __len__(self):
        return len(self.im_files)

    def __getitem__(self, index):
        filename = self.im_files[index]
        with Image.open(filename) as image:
            image = to_tensor(image.convert('RGB'))
        h, w = image.shape[-2:]
        annotation = self.labels[index]
        boxes = ops.xywh2xyxy(torch.from_numpy(annotation['bboxes'].copy()))
        boxes *= boxes.new_tensor([w, h, w, h])
        if self.train and random.random() < 0.5:
            image, boxes = flip_image_target(image, boxes)
        return image, dict(boxes=boxes, labels=torch.from_numpy(
            annotation['cls'].copy()).reshape(-1).long()), filename, (h, w)


def move_baseline_batch(batch, device):
    # Already float32 and normalized exactly once. No YOLO /255 or letterbox.
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device, non_blocking=device.type == 'cuda')
    assert batch['img'].dtype == torch.float32
    return batch


def display_batch(batch):
    # Plotting only: invert ImageNet normalization, never change training inputs.
    result = dict(batch)
    image = batch['img']
    result['img'] = (image * image.new_tensor(BASELINE_STD)[None, :, None, None]
                     + image.new_tensor(BASELINE_MEAN)[None, :, None, None]).clamp(0, 1)
    return result


class BaselineInputValidator(DetectionValidator):
    def init_metrics(self, model):
        super().init_metrics(model)
        from torchmetrics.detection.mean_ap import MeanAveragePrecision
        self.common_map = MeanAveragePrecision(box_format='xyxy', iou_type='bbox',
            max_detection_thresholds=[1, 10, 100])

    def update_metrics(self, preds, batch):
        super().update_metrics(preds, batch)
        for index, pred in enumerate(preds):
            truth = self._prepare_batch(index, batch)
            scaled = self.scale_preds(pred, truth)
            gt = self.scale_preds(dict(bboxes=truth['bboxes']), truth)
            self.common_map.update([dict(boxes=scaled['bboxes'].detach().cpu(),
                labels=scaled['cls'].long().detach().cpu(), scores=scaled['conf'].detach().cpu())],
                [dict(boxes=gt['bboxes'].detach().cpu(), labels=truth['cls'].long().detach().cpu())])

    def get_stats(self):
        stats = super().get_stats()
        common = self.common_map.compute()
        stats['metrics/mAP50(B)'] = float(common['map_50'])
        stats['metrics/mAP50-95(B)'] = float(common['map'])
        stats['fitness'] = float(common['map'])
        return stats

    def preprocess(self, batch):
        assert not self.args.half, 'Use FP32 for this protocol'
        return move_baseline_batch(batch, self.device)

    def build_dataset(self, img_path, mode='val', batch=None):
        return BaselineInputDataset(img_path, train=False)

    def scale_preds(self, pred, pbatch):
        # Exact per-axis inverse of torchvision's resize, no centered padding.
        result = {key: value.clone() for key, value in pred.items()}
        sy, sx = pbatch['ratio_pad'][0]
        result['bboxes'][:, [0, 2]] /= sx
        result['bboxes'][:, [1, 3]] /= sy
        ops.clip_boxes(result['bboxes'], pbatch['ori_shape'])
        return result

    def plot_val_samples(self, batch, ni):
        return super().plot_val_samples(display_batch(batch), ni)

    def plot_predictions(self, batch, preds, ni):
        return super().plot_predictions(display_batch(batch), preds, ni)


class AgreedYoloTrainer(DetectionTrainer):
    def build_dataset(self, img_path, mode='train', batch=None):
        return BaselineInputDataset(img_path, train=mode == 'train')

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode='train'):
        dataset = self.build_dataset(dataset_path, mode)
        return build_dataloader(dataset, batch=int(self.args.batch),
                                workers=self.args.workers, shuffle=mode == 'train', rank=rank)

    def preprocess_batch(self, batch):
        return move_baseline_batch(batch, self.device)

    def build_optimizer(self, model, name='SGD', lr=0.005, momentum=0.9,
                        decay=0.0005, iterations=1e5):
        assert name == 'SGD'
        # Match baseline: one group, decay on every trainable parameter, no Nesterov.
        return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
            lr=lr, momentum=momentum, weight_decay=decay, nesterov=False)

    def _setup_scheduler(self):
        # Native trainer expects an epoch scheduler, but the actual LR is set
        # below before EVERY optimizer update, using the shared wall-time rule.
        self.lf = lambda epoch: 1.0
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1.0)

    def optimizer_step(self):
        set_budget_lr(self.optimizer, time.time() - self.train_time_start,
                      self.args.time * 3600)
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        # Baseline has no EMA. Retain Ultralytics' checkpoint container, but copy
        # current weights instead of averaging them. Validator uses these weights.
        if self.ema:
            self.ema.ema.load_state_dict(self.model.state_dict())
            self.ema.updates += 1

    def get_validator(self):
        self.loss_names = ('box_loss', 'cls_loss', 'dfl_loss')
        return BaselineInputValidator(self.test_loader, save_dir=self.save_dir,
                                      args=copy(self.args), _callbacks=self.callbacks)

    def plot_training_samples(self, batch, ni):
        return super().plot_training_samples(display_batch(batch), ni)


@torch.inference_mode()
def predict_with_baseline_input(core_model, pil_images, device='cuda',
                                score_floor=0.001, nms_iou=0.7, max_det=100):
    """Inference adapter for YOLO(...).model, returning original-pixel boxes.

    Use instead of YOLO.predict() for these newly trained checkpoints.
    Includes tensor conversion, shared preprocessing and model postprocessing.
    """
    from ultralytics.utils import nms
    tensors = [to_tensor(image.convert('RGB')) for image in pil_images]
    original_shapes = [tuple(image.shape[-2:]) for image in tensors]
    transformed, _ = input_transform()(tensors, None)
    inputs = transformed.tensors.to(device)
    raw = core_model(inputs)
    outputs = nms.non_max_suppression(raw, conf_thres=score_floor,
        iou_thres=nms_iou, nc=2, multi_label=True, max_det=max_det)
    result = []
    for output, (h0, w0), (h1, w1) in zip(outputs, original_shapes, transformed.image_sizes):
        boxes = output[:, :4].clone()
        boxes[:, [0, 2]] *= w0 / w1
        boxes[:, [1, 3]] *= h0 / h1
        ops.clip_boxes(boxes, (h0, w0))
        result.append(dict(boxes=boxes, scores=output[:, 4], labels=output[:, 5].long()))
    return result


def check_baseline_equivalence():
    # Deliberately include rectangular inputs and an empty target.
    generator = torch.Generator().manual_seed(42)
    tensors = [torch.rand((3, 173, 311), generator=generator),
               torch.rand((3, 487, 199), generator=generator)]
    targets = [dict(boxes=torch.tensor([[11., 7., 89., 142.]]), labels=torch.tensor([0])),
               dict(boxes=torch.empty((0, 4)), labels=torch.empty((0,), dtype=torch.long))]
    samples = [(image, target, f'synthetic-{i}', tuple(image.shape[-2:]))
               for i, (image, target) in enumerate(zip(tensors, targets))]
    actual = collate_baseline_input(samples)
    # An actual Faster R-CNN model's transform, with its head/network not needed.
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    reference_model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                                            min_size=640, max_size=640)
    reference, transformed_targets = reference_model.transform(tensors, targets)
    torch.testing.assert_close(actual['img'], reference.tensors, rtol=0, atol=0)
    recovered = ops.xywh2xyxy(actual['bboxes']) * actual['bboxes'].new_tensor(
        [actual['img'].shape[-1], actual['img'].shape[-2]] * 2)
    torch.testing.assert_close(recovered, transformed_targets[0]['boxes'], atol=1e-4, rtol=1e-6)
    reflected, boxes = flip_image_target(tensors[0], targets[0]['boxes'])
    torch.testing.assert_close(reflected, tensors[0].flip(-1), rtol=0, atol=0)
    torch.testing.assert_close(boxes, torch.tensor([[222., 7., 300., 142.]]), rtol=0, atol=0)
    restored_image, restored_boxes = flip_image_target(reflected, boxes)
    torch.testing.assert_close(restored_image, tensors[0], rtol=0, atol=0)
    torch.testing.assert_close(restored_boxes, targets[0]['boxes'], rtol=0, atol=0)
    # Optimizer updates match baseline SGD. Time schedule is the new shared rule.
    parameter = torch.nn.Parameter(torch.ones(2))
    reference_parameter = torch.nn.Parameter(torch.ones(2))
    adapter = object.__new__(AgreedYoloTrainer)
    tiny = torch.nn.ParameterList([parameter])
    adapter.optimizer = adapter.build_optimizer(tiny)
    reference_optimizer = torch.optim.SGD([reference_parameter], lr=0.005,
        momentum=0.9, weight_decay=0.0005, nesterov=False)
    for elapsed in [0, 3600, 5400, 9000, 10800]:
        lr = budget_cosine_lr(elapsed, initial_lr=0.005, minimum_lr=1e-6)
        for group in adapter.optimizer.param_groups:
            group['lr'] = lr
        for group in reference_optimizer.param_groups:
            group['lr'] = lr
        assert adapter.optimizer.param_groups[0]['lr'] == reference_optimizer.param_groups[0]['lr']
        parameter.grad = torch.ones(2)
        reference_parameter.grad = torch.ones(2)
        adapter.optimizer.step()
        reference_optimizer.step()
        torch.testing.assert_close(parameter, reference_parameter, rtol=0, atol=0)
        assert lr >= 1e-6
    print('Input tensors, boxes, flip, SGD updates and budget schedule: OK')

