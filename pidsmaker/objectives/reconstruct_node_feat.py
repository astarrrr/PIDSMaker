import torch.nn as nn


class NodeFeatReconstruction(nn.Module):
    def __init__(self, decoder, loss_fn):
        super(NodeFeatReconstruction, self).__init__()
        self.decoder = decoder
        self.loss_fn = loss_fn

    def forward(self, h, x, inference, **kwargs):
        if isinstance(x, (tuple, list)):
            x = 0.5 * (x[0] + x[1])
        x_hat = self.decoder(h)
        loss = self.loss_fn(x_hat, x, inference=inference)
        return {"loss": loss}
