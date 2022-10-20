import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backbone.r2p1d import ResNet, get_r2p1d_model
from torchvision.models.video.resnet import VideoResNet
from utils.main import main, parse_args
import pytest
from utils.conf import base_path_dataset as base_path


# @pytest.mark.parametrize('dataset', ['seq-mnist', 'seq-cifar10', 'seq-cifar100', 'seq-tinyimg', 'rot-mnist', 'perm-mnist', 'mnist-360'])
# def test_der(dataset):
#     sys.argv = ['mammoth',
#                 '--model',
#                 'der',
#                 '--dataset',
#                 dataset,
#                 '--buffer_size',
#                 '10',
#                 '--lr',
#                 '1e-4',
#                 '--alpha',
#                 '.5',
#                 '--n_epochs',
#                 '1',
#                 '--debug_mode',
#                 '1']
#     a = parse_args()

#     main(a)

@pytest.mark.parametrize('fine_tune', ["fc","layer4","layer3","layer2","layer1","all"])
@pytest.mark.parametrize('model', ['R2P1_50_K700_M', 'R2P1_34_IG65_K', 'R2P1_18_K400'])
def test_r2p1d_backbone(model, fine_tune):
    m = get_r2p1d_model(model_conf=model, num_classes=2, learner_layers=3, fine_tune_up_to=fine_tune, checkpoint_path=base_path() + "checkpoints")
    assert isinstance(m, (ResNet, VideoResNet))
