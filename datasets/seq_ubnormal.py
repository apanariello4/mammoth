# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
from typing_extensions import Self
import warnings
from argparse import Namespace
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as transforms
from backbone.i3d import I3D_ResNet, i3d_resnet
from backbone.r2p1d import get_r2p1d_model
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.datasets.video_utils import VideoClips

from datasets.utils.continual_dataset import ContinualDataset
from datasets.utils.validation import get_train_val
from utils.conf import base_path_dataset as base_path
from datasets.transforms.video_transforms import ConvertBCHWtoCBHW


class UBnormal(VisionDataset):
    def __init__(
        self,
        root: str,
        frames_per_clip: int = 32,
        step_between_clips: int = 32,
        frame_rate: Optional[int] = None,
        scene: int = -1,
        train: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 4,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        download: bool = True,
    ) -> None:
        super().__init__(root)
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips
        self.frame_rate = frame_rate
        self.train = train
        self.transform = transform
        self.root = root + "/UBNORMAL" if root else base_path() + "/UBNORMAL"

        if download:
            self._load_obj_names()
            self._download_split()
            self._download_dataset() if not self._check_integrity() else None

        self.frame_level_gt = {}
        for vid in list(Path(self.root, "frame_level_gt").glob("*.txt")):
            self.frame_level_gt[vid.stem] = [int(l) for l in Path(vid).read_text().split()]
        self.test_video_names = []
        with open(Path(self.root, "split/abnormal_test_video_names.txt"), "r") as f:
            self.test_video_names += f.read().splitlines()
        with open(Path(self.root, "split/normal_test_video_names.txt"), "r") as f:
            self.test_video_names += f.read().splitlines()
        if not train:
            video_list = [str(x) for x in Path(self.root).rglob("*.mp4") if x.stem in self.test_video_names]
        else:
            video_list = [str(x) for x in Path(self.root).rglob("*.mp4") if x.stem not in self.test_video_names]

        if scene > 0:
            video_list = [x for x in video_list if f"Scene{scene}" in x]
        elif scene < 0:
            """Loads all videos up to scene <abs(scene)>"""
            video_list = [x for x in video_list if any(f"Scene{d}/" in x for d in range(1, abs(scene) + 1))]

        H = self.get_hash(video_list)
        precomputed_metadata = Path(root, "metadata", f"{H}.pt")

        metadata = torch.load(precomputed_metadata) if precomputed_metadata.exists() else None

        self.video_clips = VideoClips(
            video_list,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            num_workers=num_workers,
            _precomputed_metadata=metadata,
            output_format="TCHW"
        )

        if metadata is None:
            precomputed_metadata.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.video_clips.metadata, precomputed_metadata)
            print(f"Saved metadata for Scene {scene if scene > 0 else f'up to {abs(scene)}'} - {'Train' if train else ' Test'}")
        else:
            print(f"[Scene {scene if scene > 0 else f'up to {abs(scene)}'} - {'Train' if train else ' Test'}] Using precomputed metadata")

    def __getitem__(self, index: int) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            video_raw, _, info, video_idx = self.video_clips.get_clip(index)
        video_path = self.video_clips.video_paths[video_idx]
        if "abnormal" not in video_path:
            label = torch.tensor(0.0, dtype=torch.int64)
        else:
            _, clip_idx = self.video_clips.get_clip_location(index)
            frame_range = (clip_idx * self.frames_per_clip, (clip_idx + 1) * self.frames_per_clip)
            label = torch.tensor(1.0, dtype=torch.int64) if any(self.frame_level_gt[Path(video_path).stem][frame_range[0]:frame_range[1]]) else torch.tensor(0.0, dtype=torch.int64)

        video_raw = torchvision.transforms.Resize((112, 112))(video_raw)

        if self.transform is not None:
            video: Tensor = self.transform(video_raw)
        else:
            video: Tensor = video_raw

        if self.train:
            return video, label, video_raw
        return video, label

    def __len__(self) -> int:
        return self.video_clips.num_clips()

    def _load_obj_names(self) -> None:
        f_path = Path(self.root, "object_names_per_video.pkl")
        if not f_path.exists():
            print("UBNormal object names not found, downloading...")
            import requests
            url = "https://github.com/lilygeorgescu/UBnormal/blob/main/scripts/object_names_per_video.pkl?raw=true"
            r = requests.get(url, allow_redirects=True)
            f_path.parent.mkdir(parents=True, exist_ok=True)
            Path(f_path).write_bytes(r.content)

        import pickle

        with open(f_path, "rb") as f:
            self.obj_names_per_video = pickle.load(f)

    def _download_split(self) -> None:
        split_path = Path(self.root, "split")
        if not split_path.exists():
            split_path.mkdir(parents=True, exist_ok=True)
            print("UBNormal split not found, downloading...")
            import requests
            files = ["abnormal_test", "abnormal_train", "abnormal_validation",
                     "normal_test", "normal_train", "normal_validation"]
            for f in files:
                url = f"https://github.com/lilygeorgescu/UBnormal/blob/main/scripts/{f}_video_names.txt?raw=true"
                r = requests.get(url, allow_redirects=True)
                Path(split_path, f"{f}_video_names.txt").write_bytes(r.content)

    def _download_dataset(self) -> None:
        url = "https://unimore365-my.sharepoint.com/:u:/g/personal/265925_unimore_it/EbN91bJXnl5CnDo9U3aFIGsB5BgNdWwzYaPYEk7vID_OTA?e=iieGGj"
        from onedrivedownloader import download
        print("UBNormal dataset not found, downloading...")
        download(url, filename=f'{self.root}/UBNORMAL.zip', unzip=True, unzip_path=self.root, clean=True)

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset structure.

        Returns:
            bool: True if all the dataset directories are found, else False
        """
        return not any(not Path(self.root, f"Scene{d}").exists() for d in range(1, 30))

    def get_hash(self, video_list: List[str]) -> str:
        """Metadata is cached to avoid recomputing it every time.
           Since it can change if one changes the video list, or the
           frames_per_clip, step_between_clips, or frame_rate parameters,
           we compute a hash of all the parameters and the video list to
           uniquely identify the metadata.

        Args:
            video_list (List[str]): List of video paths.

        Returns:
            str: Hash of the parameters and the video list.
        """
        v = tuple(sorted(video_list))
        h = hashlib.sha256()
        h.update(str(v).encode())
        h.update(str(self.frames_per_clip).encode())
        h.update(str(self.step_between_clips).encode())
        h.update(str(self.frame_rate).encode())
        return h.hexdigest()[:16]


class SequenceUBnormal(ContinualDataset):
    NAME = 'seq-ubnormal'
    SETTING = 'domain-il'
    N_CLASSES_PER_TASK = 2
    N_TASKS = 10
    NO_PREFETCH = True

    def get_data_loaders(self, scene: Optional[int] = None) -> Tuple[DataLoader, DataLoader]:
        curr_scene = scene if scene is not None else self.i + 1
        transform = self.get_transform()

        test_transform = self.get_transform(train=False)

        train_dataset = UBnormal(root=self.args.data_path, scene=curr_scene, transform=transform)

        if self.args.validation:
            # TODO
            raise NotImplementedError
            train_dataset, test_dataset = get_train_val(train_dataset,
                                                        test_transform, self.NAME)
        else:
            test_dataset = UBnormal(root=self.args.data_path, train=False, transform=test_transform, scene=curr_scene)

        train, test = self.store_loaders(train_dataset, test_dataset)

        return train, test

    def get_joint_dataloaders(self):

        return self.get_data_loaders(scene=-self.N_TASKS)

    def get_backbone(self) -> torch.nn.Module:
        checkpoint_path = self.args.data_path if self.args.data_path else "/data"
        return get_r2p1d_model(model_conf="R2P1_50_K700_M", num_classes=2, learner_layers=3,
                               fine_tune_up_to="layer3", checkpoint_path=checkpoint_path + "/checkpoints")

    def store_loaders(self, train_dataset, test_dataset):

        train_loader = DataLoader(train_dataset,
                                  batch_size=self.args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        test_loader = DataLoader(test_dataset,
                                 batch_size=self.args.batch_size, shuffle=False, num_workers=4, drop_last=True)

        self.test_loaders.append(test_loader)
        self.train_loader = train_loader

        self.i += 1
        return train_loader, test_loader

    @staticmethod
    def get_epochs() -> int:
        return 10

    @staticmethod
    def get_normalization_transform():
        mean = (0.43216, 0.394666, 0.37645)
        std = (0.22803, 0.22145, 0.216989)
        return transforms.Normalize(mean, std)

    @staticmethod
    def get_batch_size() -> int:
        return 2

    def get_minibatch_size(self) -> int:
        return SequenceUBnormal.get_batch_size()

    @staticmethod
    def get_loss() -> torch.nn.Module:
        return torch.nn.CrossEntropyLoss()

    def get_transform(self, train: bool = True) -> transforms.Compose:

        crop = transforms.RandomCrop(224) if train else transforms.CenterCrop(224)

        transform = transforms.Compose([
            transforms.Resize(256),
            crop,
            transforms.ConvertImageDtype(torch.float32),
            self.get_normalization_transform(),
            ConvertBCHWtoCBHW(),
        ])

        return transform

    @staticmethod
    def get_scheduler(model, args: Namespace) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Returns the scheduler to be used for to the current dataset.
        """
        params = []
        fine_tune = model.net.fine_tune_up_to
        if fine_tune > 0:
            params.append({"params": model.net.fc.parameters()})
            for i in range(fine_tune, 5):
                params.append({"params": getattr(model.net, f'layer{i}').parameters()})

        else:
            params.append({"params": model.net.parameters()})

        model.opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.optim_wd)
        return torch.optim.lr_scheduler.MultiStepLR(model.opt, milestones=[10], gamma=0.1)
