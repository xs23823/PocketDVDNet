import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Deep convolutional stem with more layers and channels
        self.conv1 = nn.Conv2d(1, 128, 3, 1, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, 3, 1, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 256, 3, 1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, 1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 512, 3, 1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 1024, 3, 1, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.conv7 = nn.Conv2d(1024, 1024, 3, 1, padding=1)
        self.bn7 = nn.BatchNorm2d(1024)

        # Squeeze-and-Excitation block
        self.se_fc1 = nn.Linear(1024, 256)
        self.se_fc2 = nn.Linear(256, 1024)

        # Dropout and stochastic depth (simulated with Dropout)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.4)
        self.dropout3 = nn.Dropout(0.5)
        self.dropout4 = nn.Dropout(0.5)
        self.dropout5 = nn.Dropout(0.5)

        # Adaptive pooling for flexibility
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))

        # Large fully connected head
        self.fc1 = nn.Linear(1024 * 2 * 2, 2048)
        self.bn_fc1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn_fc2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn_fc3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 10)

    def forward(self, x):
        # Deep convolutional stem with residual connections
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout2(x)
        x = F.max_pool2d(x, 2)

        # Adjust residual channels to match x if necessary
        residual = x
        if residual.shape[1] != 512:
            weight = torch.eye(512, residual.shape[1], device=residual.device, dtype=residual.dtype).view(512, residual.shape[1], 1, 1)
            residual = F.conv2d(residual, weight=weight)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = x + residual  # Residual connection
        x = self.dropout3(x)
        x = F.max_pool2d(x, 2)

        residual = x
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        # Adjust residual channels to match x if necessary
        if residual.shape[1] != x.shape[1]:
            conv1x1 = nn.Conv2d(residual.shape[1], x.shape[1], kernel_size=1).to(residual.device)
            residual = conv1x1(residual)
        x = x + residual  # Residual connection

        # Squeeze-and-Excitation block
        b, c, _, _ = x.size()
        se = F.adaptive_avg_pool2d(x, 1).view(b, c)
        se = F.relu(self.se_fc1(se))
        se = torch.sigmoid(self.se_fc2(se)).view(b, c, 1, 1)
        x = x * se

        x = self.avgpool(x)
        x = self.dropout4(x)
        x = torch.flatten(x, 1)

        # Fully connected head with normalization and dropout
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout5(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = F.relu(self.bn_fc3(self.fc3(x)))
        x = self.fc4(x)

        output = F.log_softmax(x, dim=1)
        return output