import os
from collections import Counter

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class ImagenetValidationDataset(Dataset):

    def __init__(self, val_path):
        super().__init__()
        self.val_path = val_path
        self.transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])
        with open(os.path.join(val_path, 'ILSVRC2012_validation_ground_truth.txt')) as f:
            self.labels = [int(l) - 1 for l in f.readlines()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.val_path, f'ILSVRC2012_val_{item + 1:08d}.JPEG')).convert('RGB')
        return self.transform(img), self.labels[item]


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t().cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum().item()
        res.append(correct_k)
    return res


def eval_fn(net, data_loader, device='cuda'):
    net = net.to(device).eval()
    preds = []
    target = []
    for images, labels in data_loader:
        fet = net.encoder(images.to(device))
        _, pred = net.classifier(fet).topk(1, dim=1)
        preds.append(pred.squeeze(1).cpu())
        target.append(labels)
    p = torch.cat(preds).numpy()
    t = torch.cat(target).numpy()
    all_counters = [Counter() for i in range(1000)]
    for i in range(50000):
        all_counters[t[i]][p[i]] += 1
    total_correct = 0
    for i in range(1000):
        total_correct += all_counters[i].most_common(1)[0][1]
    print(f'ACC: {total_correct / 50000 * 100}')
