# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import List, Tuple

import numpy as np
import torch
from datasets.utils.validation import ValidationDataset
from sklearn.metrics import roc_auc_score
from torch.optim import SGD
from torchvision import transforms

from models.utils.continual_model import ContinualModel
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from utils.status import progress_bar


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Joint training: a strong, simple baseline.')
    add_management_args(parser)
    add_experiment_args(parser)
    return parser


class Joint(ContinualModel):
    NAME = 'joint'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(Joint, self).__init__(backbone, loss, args, transform)
        self.old_data = []
        self.old_labels = []
        self.current_task = 0

    def end_task(self, dataset):
        if dataset.SETTING != 'domain-il':
            self.old_data.append(dataset.train_loader.dataset.data)
            self.old_labels.append(torch.tensor(dataset.train_loader.dataset.targets))
            self.current_task += 1

            # # for non-incremental joint training
            if len(dataset.test_loaders) != dataset.N_TASKS:
                return

            # reinit network
            self.net = dataset.get_backbone()
            self.net.to(self.device)
            self.net.train()
            self.opt = SGD(self.net.parameters(), lr=self.args.lr)

            # prepare dataloader
            all_data, all_labels = None, None
            for i in range(len(self.old_data)):
                if all_data is None:
                    all_data = self.old_data[i]
                    all_labels = self.old_labels[i]
                else:
                    all_data = np.concatenate([all_data, self.old_data[i]])
                    all_labels = np.concatenate([all_labels, self.old_labels[i]])

            transform = dataset.TRANSFORM if dataset.TRANSFORM is not None else transforms.ToTensor()
            temp_dataset = ValidationDataset(all_data, all_labels, transform=transform)
            loader = torch.utils.data.DataLoader(temp_dataset, batch_size=self.args.batch_size, shuffle=True)

            # train
            for e in range(self.args.n_epochs):
                for i, batch in enumerate(loader):
                    inputs, labels = batch
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    self.opt.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.loss(outputs, labels.long())
                    loss.backward()
                    self.opt.step()
                    progress_bar(i, len(loader), e, 'J', loss.item())
        else:
            if not hasattr(dataset, "NO_PREFETCH"):
                self.old_data.append(dataset.train_loader)
                # train
                if len(dataset.test_loaders) != dataset.N_TASKS:
                    return

                all_inputs = []
                all_labels = []
                for source in self.old_data:
                    for x, l, _ in source:
                        all_inputs.append(x)
                        all_labels.append(l)
                all_inputs = torch.cat(all_inputs)
                all_labels = torch.cat(all_labels)
                bs = self.args.batch_size
                scheduler = dataset.get_scheduler(self, self.args)

                for e in range(self.args.n_epochs):
                    order = torch.randperm(len(all_inputs))
                    for i in range(int(math.ceil(len(all_inputs) / bs))):
                        inputs = all_inputs[order][i * bs: (i + 1) * bs]
                        labels = all_labels[order][i * bs: (i + 1) * bs]
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.opt.zero_grad()
                        outputs = self.net(inputs)
                        loss = self.loss(outputs, labels.long())
                        loss.backward()
                        self.opt.step()
                        progress_bar(i, int(math.ceil(len(all_inputs) / bs)), e, 'J', loss.item())

                    if scheduler is not None:
                        scheduler.step()
            else:

                # Domain IL and no Prefetch (Ubnormal)
                bs = self.args.batch_size
                scheduler = dataset.get_scheduler(self, self.args)
                train_loader, test_loader = dataset.get_joint_dataloaders()

                dataset.test_loaders = [test_loader]
                results, results_mask_classes = [], []
                for e in range(self.args.n_epochs):
                    correct, total = 0.0, 0.0
                    for i, (inputs, labels, _) in enumerate(train_loader):
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        self.opt.zero_grad()
                        outputs = self.net(inputs)
                        loss = self.loss(outputs, labels.long())
                        loss.backward()
                        self.opt.step()
                        progress_bar(i, len(train_loader), e, 'J', loss.item())
                        correct += torch.sum(torch.max(outputs.data, 1)[1] == labels).item()
                        total += labels.shape[0]

                    print(f"\n[{e}] Train Accuracy: ", correct / total)
                    accs = self.evaluate_video(test_loader, e)

                    results.append(accs[0])
                    results_mask_classes.append(accs[1])

                    if scheduler is not None:
                        scheduler.step()

                return results, results_mask_classes

    def observe(self, inputs, labels, not_aug_inputs):
        return 0

    def evaluate_video(self, test_loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[List[float], List[float]]:
        self.net.eval()
        accs, accs_mask_classes = [], []
        correct, correct_mask_classes, total = 0.0, 0.0, 0.0
        all_outputs, all_labels = [], []
        for k, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs = self.net(inputs)
            loss = self.loss(outputs, labels.long())
            _, pred = torch.max(outputs.data, 1)
            all_outputs += outputs.cpu().detach().numpy().tolist()
            all_labels += labels.cpu().numpy().tolist()
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]
            progress_bar(k, len(test_loader), epoch, 'J - Test', loss.item())

        all_outputs = np.array(all_outputs)
        all_labels = np.array(all_labels)
        all_labels = np.squeeze(np.eye(outputs.shape[1])[all_labels.reshape(-1)])
        accs.append(correct / total * 100 if 'class-il' in self.COMPATIBILITY else 0)
        accs_mask_classes.append(correct_mask_classes / total * 100)
        print("\nRoc macro: ", roc_auc_score(all_labels, all_outputs, average='macro'))
        print("Roc micro: ", roc_auc_score(all_labels, all_outputs, average='micro'))
        print("Accuracy: ", correct / total * 100)
        print("Accuracy mask classes: ", correct_mask_classes / total * 100)
        self.net.train()

        return accs, accs_mask_classes
