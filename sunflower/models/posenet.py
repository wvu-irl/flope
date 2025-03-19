import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class PoseResNet(nn.Module):
    def __init__(self, backbone_out_dim=2048, dropout=0.5):
        super(PoseResNet, self).__init__()
        
        #! Backbone
        self.base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        fc_in_features = self.base.fc.in_features
        self.base.avgpool = nn.AdaptiveAvgPool2d(1)
        self.base.fc = nn.Sequential(
            nn.Linear(fc_in_features, backbone_out_dim),
            nn.ReLU()
        )
        
        #! Final Layer
        self.fc_rot = nn.Linear(backbone_out_dim, 9)
        
        #! Dropouts
        self.dropout = dropout

    def extract_features(self, x):
        x_features = self.base(x)
        x_features = F.relu(x_features)
        if self.dropout > 0:
            x_features = F.dropout(x_features, p=self.dropout, training=self.training)
        return x_features

    def forward(self, x):
        x_features = self.extract_features(x)
        x_rotations = self.fc_rot(x_features)
        return x_rotations
    
if __name__=='__main__':
    from PIL import Image
    posenet = PoseResNet().to('cuda')
    img = Image.open('/home/rashik_shrestha/data/flower_posenet_data/images/000000.png')
    print(img.size)