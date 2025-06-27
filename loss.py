import torch
from torch import nn
from torchvision.models import vgg16


class CustomGeneratorLoss(nn.Module):
    def __init__(self):
        super(CustomGeneratorLoss, self).__init__()
        vgg_model = vgg16(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg_model.features)[:31]).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.l2_loss = nn.MSELoss()
        self.tv_regularizer = TotalVariationLoss()

    def forward(self, discriminator_preds, generated_imgs, real_imgs):
        # Adversarial component
        adversarial_term = torch.mean(1 - discriminator_preds)

        # Perceptual (feature space) loss
        perceptual_term = self.l2_loss(self.feature_extractor(generated_imgs),
                                       self.feature_extractor(real_imgs))

        # Pixel-wise MSE loss
        reconstruction_term = self.l2_loss(generated_imgs, real_imgs)

        # TV regularization
        tv_term = self.tv_regularizer(generated_imgs)

        # Final weighted loss
        total_loss = reconstruction_term + 0.001 * adversarial_term + 0.006 * perceptual_term + 2e-8 * tv_term
        return total_loss


class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    def forward(self, img):
        b, c, h, w = img.size()
        count_h = self._num_elements(img[:, :, 1:, :])
        count_w = self._num_elements(img[:, :, :, 1:])
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :h - 1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :w - 1], 2).sum()
        return self.weight * 2 * (tv_h / count_h + tv_w / count_w) / b

    @staticmethod
    def _num_elements(tensor):
        return tensor.size(1) * tensor.size(2) * tensor.size(3)


if __name__ == "__main__":
    criterion = CustomGeneratorLoss()
    print(criterion)
