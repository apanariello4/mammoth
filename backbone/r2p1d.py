from collections import OrderedDict
from functools import partial
from pathlib import Path
import types

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from backbone import MammothBackbone


class Learner(nn.Module):
    def __init__(self, layers, input_dim: int, output_dim: int, drop_p: float = 0.0, activation: str = 'relu'):
        super(Learner, self).__init__()

        activation_fun: nn.Module = {"relu": nn.ReLU(inplace=True),
                                     "silu": nn.SiLU(inplace=True),
                                     "gelu": nn.GELU(),
                                     "leaky_relu": nn.LeakyReLU(inplace=True),
                                     "elu": nn.ELU(inplace=True)}[activation]
        hidden_dim = 512 if input_dim > 512 else input_dim // 2
        if layers == 2:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim, output_dim),
            )
        elif layers == 3:
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(hidden_dim, 32),
                activation_fun,
                nn.Dropout(drop_p),
                nn.Linear(32, output_dim),
            )
        else:
            raise ValueError("Only 2 or 3 layers are supported")

        self.weight_init()

    def weight_init(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.classifier(x)


def get_inplanes():
    return [64, 128, 256, 512]


def conv1x3x3(in_planes, mid_planes, stride=1):
    return nn.Conv3d(in_planes,
                     mid_planes,
                     kernel_size=(1, 3, 3),
                     stride=(1, stride, stride),
                     padding=(0, 1, 1),
                     bias=False)


def conv3x1x1(mid_planes, planes, stride=1):
    return nn.Conv3d(mid_planes,
                     planes,
                     kernel_size=(3, 1, 1),
                     stride=(stride, 1, 1),
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        n_3d_parameters1 = in_planes * planes * 3 * 3 * 3
        n_2p1d_parameters1 = in_planes * 3 * 3 + 3 * planes
        mid_planes1 = n_3d_parameters1 // n_2p1d_parameters1
        self.conv1_s = conv1x3x3(in_planes, mid_planes1, stride)
        self.bn1_s = nn.BatchNorm3d(mid_planes1)
        self.conv1_t = conv3x1x1(mid_planes1, planes, stride)
        self.bn1_t = nn.BatchNorm3d(planes)

        n_3d_parameters2 = planes * planes * 3 * 3 * 3
        n_2p1d_parameters2 = planes * 3 * 3 + 3 * planes
        mid_planes2 = n_3d_parameters2 // n_2p1d_parameters2
        self.conv2_s = conv1x3x3(planes, mid_planes2)
        self.bn2_s = nn.BatchNorm3d(mid_planes2)
        self.conv2_t = conv3x1x1(mid_planes2, planes)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1_s(x)
        out = self.bn1_s(out)
        out = self.relu(out)
        out = self.conv1_t(out)
        out = self.bn1_t(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)

        n_3d_parameters = planes * planes * 3 * 3 * 3
        n_2p1d_parameters = planes * 3 * 3 + 3 * planes
        mid_planes = n_3d_parameters // n_2p1d_parameters
        self.conv2_s = conv1x3x3(planes, mid_planes, stride)
        self.bn2_s = nn.BatchNorm3d(mid_planes)
        self.conv2_t = conv3x1x1(mid_planes, planes, stride)
        self.bn2_t = nn.BatchNorm3d(planes)

        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2_s(out)
        out = self.bn2_s(out)
        out = self.relu(out)
        out = self.conv2_t(out)
        out = self.bn2_t(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(MammothBackbone):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=400,
                 frozen_stages=-1):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        n_3d_parameters = 3 * self.in_planes * conv1_t_size * 7 * 7
        n_2p1d_parameters = 3 * 7 * 7 + conv1_t_size * self.in_planes
        mid_planes = n_3d_parameters // n_2p1d_parameters

        assert (frozen_stages in (-1, 0, 1, 2, 3, 4))
        self.frozen_stages = frozen_stages

        self.layer0 = nn.Sequential(OrderedDict([
            ('conv1_s', nn.Conv3d(n_input_channels,
                                  mid_planes,
                                  kernel_size=(1, 7, 7),
                                  stride=(1, 2, 2),
                                  padding=(0, 3, 3),
                                  bias=False)),
            ('bn1_s', nn.BatchNorm3d(mid_planes)),
            ('relu1_s', nn.ReLU(inplace=True)),
            ('conv1_t', nn.Conv3d(mid_planes,
                                  self.in_planes,
                                  kernel_size=(conv1_t_size, 1, 1),
                                  stride=(conv1_t_stride, 1, 1),
                                  padding=(conv1_t_size // 2, 0, 0),
                                  bias=False)),
            ('bn1_t', nn.BatchNorm3d(self.in_planes)),
            ('relu1_t', nn.ReLU(inplace=True)),
            ('maxpool1', nn.Sequential() if no_max_pool else nn.MaxPool3d(kernel_size=3, stride=2, padding=1))])
        )

        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=2)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=2)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.num_features = block_inplanes[3] * block.expansion
        self.fc = nn.Linear(self.num_features, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def load_state_dict_from_repo_version(self, checkpoint):
        edited_dict = checkpoint["state_dict"]
        remaped_key = {s: "layer0." + s for s in
                       list(edited_dict.keys()) if not s.startswith("layer") and not s.startswith("fc")}
        state_dict = dict((remaped_key[key], edited_dict[key]) if key in remaped_key
                          else (key, value) for key, value in edited_dict.items())

        self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes *
                              block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_features=False):

        for i in range(5):
            layer = getattr(self, f"layer{i}")
            if i <= self.frozen_stages:
                with torch.no_grad():
                    x = layer(x)
            else:
                x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if return_features:
            return x

        x = self.fc(x)

        return x

    def train(self, mode=True):
        """Convert the model into training mode while keeping layers freezed."""
        super().train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for i in range(0, self.frozen_stages + 1):
                m = getattr(self, f'layer{i}')
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False


def generate_model(model_depth, **kwargs):
    assert model_depth in (10, 18, 34, 50, 101, 152, 200)

    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


K700_MEAN_STD = ((0.4345, 0.4051, 0.3775), (0.2768, 0.2713, 0.2737))
K400_MEAN_STD = ((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989))

FINE_TUNE = {"fc": 4,
             "layer4": 3,
             "layer3": 2,
             "layer2": 1,
             "layer1": 0,
             "all": -1}


def get_r2p1d_model(model_conf: str,
                    num_classes: int,
                    learner_layers: int,
                    fine_tune_up_to: str,
                    checkpoint_path: str,
                    dropout: float = 0.6,
                    pretrained: bool = True):
    """loads the model from the config string in format
        R2P1_layers_pretrain,
        "R2P1_18_K700","R2P1_34_K700", from https://github.com/kenshohara/3D-ResNets-PyTorch,
        "R2P1_50_K700", "R2P1_50_K700_M" from https://github.com/kenshohara/3D-ResNets-PyTorch,
        "R2P1_18_K400" from torchvision
        "R2P1_34_IG65", "R2P1_34_IG65_K" from https://github.com/moabitcoin/ig65m-pytorch,
        "R2P1_LSTM_50_K700_M" custom model with lstm.

    Args:
        model_conf (str): model configuration string
        learner_layers (int): if 0 uses 1 fc layer with batch norm, otherwise uses
            learner with <learner_layers> layers [2, 3]
        dropout (float): dropout rate when using learner
        checkpoint_path (str, optional): path to the checkpoint files.

    Returns:
        model (nn.Module): the model
        mean (tuple): mean of the dataset
        std (tuple): std of the dataset
    """

    assert learner_layers in (1, 2, 3)

    if model_conf == "R2P1_50_K700_M":

        frozen_stages = FINE_TUNE.get(fine_tune_up_to)
        if pretrained:
            model = generate_model(50, n_classes=1039, frozen_stages=frozen_stages)
            checkpoint = load_checkpoint(f"{checkpoint_path}/r2p1d50_KM_200ep.pth")
            model.load_state_dict_from_repo_version(checkpoint)
        else:
            model = generate_model(50, n_classes=num_classes, frozen_stages=frozen_stages)
        mean, std = K700_MEAN_STD
        num_features_out = 2048

    elif model_conf == "R2P1_18_K400":
        model = torchvision.models.video.r2plus1d_18(weights='KINETICS400_V1' if pretrained else None)
        mean, std = K400_MEAN_STD
        num_features_out = 512
        freeze_stages(model, fine_tune_up_to)

    elif model_conf == "R2P1_34_IG65_K":
        model = torch.hub.load("moabitcoin/ig65m-pytorch",
                               "r2plus1d_34_32_kinetics", num_classes=400, pretrained=pretrained)
        mean, std = K400_MEAN_STD
        num_features_out = 512
        freeze_stages(model, fine_tune_up_to)
    else:
        raise NotImplementedError(f"model {model_conf} not yet supported")

    if not learner_layers:
        model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=num_features_out),
            nn.Linear(num_features_out, num_classes),
        )
    else:
        model.fc = Learner(layers=learner_layers,
                           input_dim=num_features_out,
                           output_dim=num_classes,
                           drop_p=dropout,)

    model.fine_tune_up_to = FINE_TUNE.get(fine_tune_up_to)

    return model


def freeze_stages(model, fine_tune_up_to):
    frozen_stages = FINE_TUNE.get(fine_tune_up_to)
    if frozen_stages >= 0:
        if hasattr(model, "stem"):
            for param in model.stem.parameters():
                model.stem.eval()
                param.requires_grad = False

        for i in range(1, frozen_stages + 1):
            m = getattr(model, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(self).train(mode)
        self._freeze_stages()

    model.train = types.MethodType(train, model)


def load_checkpoint(path: str):
    urls = {"r2p1d50_KM_200ep": "https://unimore365-my.sharepoint.com/:u:/g/personal/265925_unimore_it/EQ3Il_kGVlJCg5dxizIdaaUBUsvg8nd5ZenRya6_M3MtGA?e=SihgW2"}
    if not Path(path).exists():
        from onedrivedownloader import download
        print(f"Downloading {path} checkpoint")
        download(urls[Path(path).stem], path, unzip=False)

    checkpoint = torch.load(path, map_location="cpu")
    return checkpoint


if __name__ == "__main__":
    pass
