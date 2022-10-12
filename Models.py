import torch
import torch.nn as nn

from myXception import Xception_Network
from Inception import InceptionResnetV2


class Mixure_Model(nn.Module):
    def __init__(self, height, width, num_classes):
        super(Mixure_Model, self).__init__()
        self.name = 'Mixture'

        self.Incept_extract = InceptionResnetV2(height, width, num_classes)
        self.Xcept_extract = Xception_Network(height, width, num_classes)
        self.feature_dim = 1536 + 2048
        self.ln = nn.Linear(self.feature_dim, num_classes)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = torch.cat((self.Incept_extract(x), self.Xcept_extract(x)), 1)
        x = self.ln(x)
        x = self.sm(x)
        return x



class Xception_Model(nn.Module):
    def __init__(self, height, width, num_classes):
        super(Xception_Model, self).__init__()
        self.name = 'Xception'
        self.size = (height, width)

        self.Feature_extract = Xception_Network(height, width, num_classes)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.Feature_extract(x)
        x = self.sm(x)
        return x



class Inception_Model(nn.Module):
    def __init__(self, height, width, num_classes):
        super(Inception_Model, self).__init__()
        self.name = 'Inception'
        self.size = (height, width)

        self.Feature_extract = InceptionResnetV2(height, width, num_classes)
        self.sm = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.Feature_extract(x)
        x = self.sm(x)
        return x

