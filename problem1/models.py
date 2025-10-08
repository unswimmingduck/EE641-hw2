import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, conditional=False, num_classes=26):
        """
        Generator network that produces 28×28 letter images.

        Args:
            z_dim: Dimension of latent vector z
            conditional: If True, condition on letter class
            num_classes: Number of letter classes (26)
        """
        super().__init__()
        self.z_dim = z_dim
        self.conditional = conditional
        self.num_classes = num_classes

        input_dim = z_dim + (num_classes if conditional else 0)

        self.project = nn.Sequential(
            nn.Linear(input_dim, 128 * 7 * 7),
            nn.BatchNorm1d(128 * 7 * 7),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            nn.Unflatten(1, (128, 7, 7)),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),  
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # output in [-1, 1]
        )

    def forward(self, z, class_label=None):
        """
        Generate images from latent code.

        Args:
            z: Latent vectors [batch_size, z_dim]
            class_label: One-hot encoded class labels [batch_size, num_classes]

        Returns:
            Generated images [batch_size, 1, 28, 28] in range [-1, 1]
        """
        if self.conditional:
            if class_label is None:
                raise ValueError("Conditional=True requires class_label one-hot vectors")
            x = torch.cat([z, class_label], dim=1)
        else:
            x = z
        x = self.project(x)
        img = self.main(x)
        return img

class Discriminator(nn.Module):
    def __init__(self, conditional=False, num_classes=26):
        """
        Discriminator network that classifies 28×28 images as real/fake.
        """
        super().__init__()
        self.conditional = conditional
        self.num_classes = num_classes

        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),   
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),

            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.25),
        )

        feature_dim = 256 * 3 * 3
        clf_in = feature_dim + (num_classes if conditional else 0)
        self.classifier = nn.Sequential(
            nn.Linear(clf_in, 1),
            nn.Sigmoid()
        )

    def extract_features(self, img):
        return torch.flatten(self.features(img), 1)

    def forward(self, img, class_label=None):
        """
        Classify images as real (1) or fake (0).

        Returns:
            Probability of being real [batch_size, 1]
        """
        feat = self.extract_features(img)
        if self.conditional:
            if class_label is None:
                raise ValueError("Conditional=True requires class_label one-hot vectors")
            feat = torch.cat([feat, class_label], dim=1)
        out = self.classifier(feat)
        return out
