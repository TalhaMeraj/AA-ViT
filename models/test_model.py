from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      opt.norm, not opt.no_dropout,
                                      opt.init_type,
                                      self.gpu_ids)
        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def get_rdeb_heatmap(self, layer='encoder_1'):
        """
        Extract mean-absolute heatmap from a given RDEB layer.
        layer: 'encoder_1', 'encoder_2', 'encoder_3'
        Returns: heatmap [H,W] normalized to [0,1]
        """
        with torch.no_grad():
            if layer == 'encoder_1':
                x = self.netG.encoder_1[0:4](self.real_A)  # pad+conv+bn+ReLU
                rdeb_out = self.netG.encoder_1[4](x)
            elif layer == 'encoder_2':
                x = self.netG.encoder_1(self.real_A)
                x = self.netG.encoder_2[0:3](x)
                rdeb_out = self.netG.encoder_2[3](x)
            elif layer == 'encoder_3':
                x = self.netG.encoder_1(self.real_A)
                x = self.netG.encoder_2(x)
                x = self.netG.encoder_3[0:3](x)
                rdeb_out = self.netG.encoder_3[3](x)
            else:
                raise ValueError('Invalid layer name')

            heatmap = rdeb_out.abs().mean(dim=1).cpu().numpy()[0]  # average over channels
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
            return heatmap

    def compute_saliency(self, target=None):
        """
        Compute gradient-based saliency map for 3-channel input.
        target: ground-truth tensor [1,3,H,W], if None use input as pseudo-GT
        Returns: saliency [H,W] normalized
        """
        self.real_A.requires_grad_(True)
        fake_B = self.netG(self.real_A)

        if target is None:
            target = self.real_A  # pseudo-GT for unsupervised test
        loss = F.mse_loss(fake_B, target)

        grads = torch.autograd.grad(outputs=loss, inputs=self.real_A,
                                    create_graph=False, retain_graph=True)[0]
        saliency = grads.abs().mean(dim=1).detach().cpu().numpy()[0]  # average over channels
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        return saliency
