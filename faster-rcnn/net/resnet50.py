import torchvision.models  as models
import torch.nn as nn


def get_resnet():
    model = models.resnet50(pretrained=True)
    features = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    # 获取分类部分
    classifier = list([model.layer4, model.avgpool])
    features = nn.Sequential(*features)
    classifier = nn.Sequential(*classifier)

    return features, classifier
