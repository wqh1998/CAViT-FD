from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
from . import dist_model


class exportPerceptualLoss(torch.nn.Module):
    def __init__(
        self, model="net-lin", net="alex", colorspace="rgb", spatial=False, use_gpu=True
    ):  # VGG using our perceptually-learned weights (LPIPS metric)
        super(exportPerceptualLoss, self).__init__()
        print("Setting up Perceptual loss...")
        self.use_gpu = use_gpu
        self.spatial = spatial
        self.model = dist_model.exportModel()
        self.model.initialize(
            model=model,
            net=net,
            use_gpu=use_gpu,
            colorspace=colorspace,
            spatial=self.spatial,
        )
        print("...[%s] initialized" % self.model.name())
        print("...Done")

    def forward(self, pred, target):
        return self.model.forward(target, pred)
