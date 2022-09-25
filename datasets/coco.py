# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask
from util.box_ops import box_cxcywh_to_xyxy

import datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms, return_masks, large_scale_jitter, image_set):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.large_scale_jitter = large_scale_jitter
        self.image_set = image_set
        self.num_bins = 1000
        self.vocal = 1094

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            if self.large_scale_jitter and self.image_set == "train":
                img1, target1 = self._transforms(img, target)
                img2, target2 = self._transforms(img, target)
                return img1, img2, self.build_seqs(target1), self.build_seqs(target2)
            else:
                img, target = self._transforms(img, target)
                return img, self.build_seqs(target)
        return img, self.build_seqs(target)

    def build_target_seq(self, target, max_objects=100):

        label = target["labels"]
        box = target["boxes"]
        img_size = target["size"]
        h, w = img_size[0], img_size[1]

        label = label.unsqueeze(1) + self.num_bins + 1
        box = box * torch.stack([w, h, w, h], dim=0)
        box = box_cxcywh_to_xyxy(box)
        box = (box / 640 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
        target_tokens = torch.cat([box, label], dim=1).flatten()

        end_token = torch.tensor([self.num_vocal - 2], dtype=torch.int64)

        num_noise = max_objects - len(label)
        fake_target_tokens = torch.zeros((num_noise, 5), dtype=torch.int64)
        fake_target_tokens[:, :3] = -100
        fake_target_tokens[:, 3] = self.num_vocal - 1  # noise class
        fake_target_tokens[:, 4] = self.num_vocal - 2  # eos
        fake_target_tokens = fake_target_tokens.flatten()

        target_seq = torch.cat([target_tokens, end_token, fake_target_tokens], dim=0)

        return target_seq
    
    def gaussian_radius(self, det_size, min_overlap=0.7):
        height, width = det_size

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2
        return torch.min(torch.stack((r1,r2,r3)), 0).values

    def gaussian1D(self, diameter, sigma):
        radius = (diameter - 1.) / 2.
        # x = np.ogrid[-radius:radius+1]
        x = torch.linspace(-radius,radius,diameter)

        h = torch.exp(-(x * x) / (2 * sigma * sigma))
        h[h < torch.finfo(h.dtype).eps * h.max()] = 0
        return h

    def build_focal_target_seq(self, target, max_objects=100):
        label = target["labels"]
        box = target["boxes"]
        img_size = target["size"]
        h, w = img_size[0], img_size[1]

        label = label.unsqueeze(1) + self.num_bins + 1
        box = box * torch.stack([w, h, w, h], dim=0)
        box = box_cxcywh_to_xyxy(box)
        box = (box / 640 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
        width_arr  = box[:,2] - box[:,0]
        height_arr = box[:,3] - box[:,1]
        if len(label) > 0:
            radius_arr = self.gaussian_radius((height_arr, width_arr))
        # print(radius_arr.shape)

        # for object in range(box.size()[0]):
        focal_target_distributions = []
        img_size_arr = (img_size / 640 * self.num_bins).floor().long().clamp(min=0, max=self.num_bins)
        for object in range(len(label)):
            for i in range(4):
                distribution = torch.zeros(self.num_vocal)
                radius = radius_arr[object]
                radius = max(0, radius.long())
                diameter = 2 * radius + 1
                # diameter = diameter.cpu().numpy()
                gaussian = self.gaussian1D(diameter, diameter / 6)
                # gaussian = torch.from_numpy(gaussian)
                # print(gaussian)
                center = box[object][i]
                width = img_size_arr[1]
                height = img_size_arr[0]
                # print(center)
                # print(width)
                # print(radius)
                # exit(0)
                if i % 2 == 0: #x
                    low, high = min(center, radius), min(width - center, radius + 1)
                else: #y
                    low, high = min(center, radius), min(height - center, radius + 1)
                # print(center)
                # print(low)
                # print(high)
                # exit(0)
                masked_distribution  = distribution[center - low:center + high]
                masked_gaussian = gaussian[radius - low:radius + high]
                if min(masked_gaussian.shape) > 0 and min(masked_distribution.shape) > 0: 
                    distribution[center - low:center + high] = masked_gaussian
                # print(distribution)
                # print(distribution.sum())
                # exit(0)
                focal_target_distributions.append(distribution)
            distribution = torch.zeros(self.num_vocal)
            focal_target_distributions.append(distribution)
        if len(label) > 0:
            target_distributions = torch.stack(focal_target_distributions, dim=0)
        else:
            target_distributions = torch.tensor([])
        # target_tokens = torch.cat([box, label], dim=1).flatten()
        end_distribution = torch.zeros((1, self.num_vocal))
        # end_token = torch.tensor([self.num_vocal - 2], dtype=torch.int64).to(device)

        num_noise = max_objects - len(label)
        fake_target_distributions = torch.zeros((num_noise*5, self.num_vocal))
        # fake_target_tokens = torch.zeros((num_noise, 5), dtype=torch.int64).to(device)
        # fake_target_tokens[:, :3] = -100
        # fake_target_tokens[:, 3] = self.num_vocal - 1  # noise class
        # fake_target_tokens[:, 4] = self.num_vocal - 2  # eos
        # fake_target_tokens = fake_target_tokens.flatten()

        target_seq = torch.cat([target_distributions, end_distribution, fake_target_distributions], dim=0)

        return target_seq

    def build_seqs(self, target):
        # focal_target_seq = self.build_focal_target_seq(targets)
        # target_seq = self.build_target_seq(targets)
        target = target.copy()
        focal_target_seq = self.build_focal_target_seq(target)
        target_seq = self.build_target_seq(target)
        target["focal_target_seq"] = focal_target_seq
        target["target_seq"] = target_seq
        return target 




def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks

            polygons = [torch.tensor(obj["segmentation"][0]) for obj in anno]
            num_per_polygon = torch.tensor([p.shape[0] for p in polygons], dtype=torch.int64)
            new_polygons = torch.zeros([len(polygons), max(num_per_polygon)])
            for gt_i, (np, p) in enumerate(zip(num_per_polygon, polygons)):
                new_polygons[gt_i, :np] = p
            target["polygons"] = new_polygons
            target["valid_pol_idx"] = num_per_polygon

        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set, args):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        if args.large_scale_jitter:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.LargeScaleJitter(output_size=640, aug_scale_min=0.3, aug_scale_max=2.0),
                T.RandomDistortion(0.5, 0.5, 0.5, 0.5),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=640),
                    T.Compose([
                        T.RandomResize([400, 500, 600]),
                        T.RandomSizeCrop(384, 600),
                        T.RandomResize(scales, max_size=640),
                    ])
                ),
                normalize,
            ])

    if image_set == 'val':
        if args.large_scale_jitter:
            return T.Compose([
                T.LargeScaleJitter(output_size=640, aug_scale_min=1.0, aug_scale_max=1.0),
                normalize,
            ])
        else:
            return T.Compose([
                T.RandomResize([800], max_size=640),
                normalize,
            ])

    raise ValueError(f'unknown {image_set}')


def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set, args),
        return_masks=False,
        large_scale_jitter=args.large_scale_jitter,
        image_set=image_set)
    return dataset
