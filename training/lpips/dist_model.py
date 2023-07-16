from __future__ import absolute_import
import torch
import os
from . import networks_basic as networks

class exportModel(torch.nn.Module):
    def name(self):
        return self.model_name

    def initialize(
        self,
        model="net-lin",
        net="vgg",
        colorspace="Lab",
        pnet_rand=False,
        pnet_tune=False,
        model_path=None,
        use_gpu=True,
        printNet=False,
        spatial=False,
        is_train=False,
        lr=0.0001,
        beta1=0.5,
        version="0.1",
    ):

        self.model = model
        self.net = net
        self.is_train = is_train
        self.spatial = spatial
        self.use_gpu = use_gpu
        self.model_name = "%s [%s]" % (model, net)

        assert self.model == "net-lin"  # pretrained net + linear layer
        self.net = networks.PNetLin(
            pnet_rand=pnet_rand,
            pnet_tune=pnet_tune,
            pnet_type=net,
            use_dropout=True,
            spatial=spatial,
            version=version,
            lpips=True,
        )
        kw = {}
        if not use_gpu:
            kw["map_location"] = "cpu"
        if model_path is None:
            import inspect

            model_path = os.path.abspath(
                os.path.join(
                    inspect.getfile(self.initialize),
                    "..",
                    "weights/v%s/%s.pth" % (version, net),
                )
            )

        assert not is_train
        print("Loading model from: %s" % model_path)
        self.net.load_state_dict(torch.load(model_path, **kw), strict=False)
        self.net.eval()

        if printNet:
            print("---------- Networks initialized -------------")
            networks.print_network(self.net)
            print("-----------------------------------------------")

    def forward(self, in0, in1, retPerLayer=False):
        """Function computes the distance between image patches in0 and in1
        INPUTS
            in0, in1 - torch.Tensor object of shape Nx3xXxY - image patch scaled to [-1,1]
        OUTPUT
            computed distances between in0 and in1
        """

        return self.net.forward(in0, in1, retPerLayer=retPerLayer)