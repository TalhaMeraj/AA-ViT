import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from torchvision import models
import numpy as np
import torchvision.transforms as transforms
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from collections import OrderedDict
from PIL import Image
import torch.nn.functional as F
from packaging import version
import pytorch_ssim  # or use your own SSIM implementation
from torchmetrics import StructuralSimilarityIndexMeasure


def normalize_for_ssim(x):
    # Convert [-1,1] -> [0,1] for SSIM
    return (x + 1.0) / 2.0
# import sys
# sys.path.append("/home/fnu.talha/resvit/fomo/src/models/networks")
# sys.path.append("/home/fnu.talha/resvit/fomo/src/models/conv_blocks")

# from unet import UNet
# import torch.nn as nn

# def load_segmentation_unet(pretrained_path, in_ch=1, device="cuda"):
#     # Load UNet in segmentation mode
#     model = UNet(mode="segmentation", input_channels=in_ch)
#     convert_to_2d(model.encoder)  # convert 3D to 2D if needed
#     convert_to_2d(model.decoder)  # decoder needed for segmentation forward
#     ckpt = torch.load(pretrained_path, map_location=device)
#     state_dict = ckpt.get("model", ckpt)
#     model.load_state_dict(state_dict, strict=False)
#     model.to(device).eval()  # freeze model
#     return model

# def to_tuple(val, ndim=3):
#     if isinstance(val, int):
#         return (val,) * ndim
#     return val


# def normalize_features(feat):
#     dims = list(range(feat.dim()))  # e.g., [0,1,2,3] or [0,1,2,3,4]
#     dims.remove(1)  # keep channel dim intact, normalize over all other dims
#     mean = feat.mean(dim=dims, keepdim=True)
#     std = feat.std(dim=dims, keepdim=True) + 1e-6
#     return (feat - mean) / std


# def convert_to_2d(module):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Conv3d):
#             conv2d = nn.Conv2d(
#                 in_channels=child.in_channels,
#                 out_channels=child.out_channels,
#                 kernel_size=to_tuple(child.kernel_size, 3)[1:],
#                 stride=to_tuple(child.stride, 3)[1:],
#                 padding=to_tuple(child.padding, 3)[1:],
#                 bias=(child.bias is not None),
#             )
#             setattr(module, name, conv2d)
#         elif isinstance(child, nn.MaxPool3d):
#             k = to_tuple(child.kernel_size, 3)[1:]
#             s = to_tuple(child.stride, 3)[1:]
#             p = to_tuple(child.padding, 3)[1:]
#             pool2d = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
#             setattr(module, name, pool2d)
#         elif isinstance(child, nn.InstanceNorm3d):
#             norm2d = nn.InstanceNorm2d(
#                 num_features=child.num_features,
#                 eps=child.eps,
#                 momentum=child.momentum,
#                 affine=child.affine,
#                 track_running_stats=child.track_running_stats
#             )
#             setattr(module, name, norm2d)
#         else:
#             convert_to_2d(child)


# def convert_to_2d(module):
#     for name, child in module.named_children():
#         if isinstance(child, nn.Conv3d):
#             conv2d = nn.Conv2d(
#                 in_channels=child.in_channels,
#                 out_channels=child.out_channels,
#                 kernel_size=to_tuple(child.kernel_size, 3)[1:],
#                 stride=to_tuple(child.stride, 3)[1:],
#                 padding=to_tuple(child.padding, 3)[1:],
#                 bias=(child.bias is not None),
#             )
#             setattr(module, name, conv2d)

#         elif isinstance(child, nn.ConvTranspose3d):
#             convt2d = nn.ConvTranspose2d(
#                 in_channels=child.in_channels,
#                 out_channels=child.out_channels,
#                 kernel_size=to_tuple(child.kernel_size, 3)[1:],
#                 stride=to_tuple(child.stride, 3)[1:],
#                 padding=to_tuple(child.padding, 3)[1:],
#                 output_padding=to_tuple(child.output_padding, 3)[1:],
#                 bias=(child.bias is not None),
#             )
#             setattr(module, name, convt2d)

#         elif isinstance(child, nn.MaxPool3d):
#             k = to_tuple(child.kernel_size, 3)[1:]
#             s = to_tuple(child.stride, 3)[1:]
#             p = to_tuple(child.padding, 3)[1:]
#             pool2d = nn.MaxPool2d(kernel_size=k, stride=s, padding=p)
#             setattr(module, name, pool2d)

#         elif isinstance(child, nn.InstanceNorm3d):
#             norm2d = nn.InstanceNorm2d(
#                 num_features=child.num_features,
#                 eps=child.eps,
#                 momentum=child.momentum,
#                 affine=child.affine,
#                 track_running_stats=child.track_running_stats
#             )
#             setattr(module, name, norm2d)

#         else:
#             convert_to_2d(child)


# def get_segmentation_features(seg_model, x):
#     feats = {}
    
#     # Encoder features
#     stages = [
#         # ("x0", seg_model.encoder.in_conv),
#         ("x1", seg_model.encoder.encoder_conv1),
#         ("x2", seg_model.encoder.encoder_conv2),
#         # ("x3", seg_model.encoder.encoder_conv3),
#         # ("x4", seg_model.encoder.encoder_conv4),
#     ]
    
#     def hook_fn(name):
#         def fn(module, input, output):
#             feats[name] = output
#         return fn

#     handles = []
#     for name, layer in stages:
#         handles.append(layer.register_forward_hook(hook_fn(name)))

#     # Forward pass (includes encoder -> decoder)
#     _ = seg_model(x)

#     for h in handles:
#         h.remove()
    
#     # Optionally add the final segmentation output
#     # feats["logits"] = seg_model(x) if isinstance(seg_model(x), torch.Tensor) else seg_model(x)[0]
    
#     return feats


# def load_encoder(pretrained_path, in_ch=1, device="cuda"):  # <-- default to 1
#     model = UNet(mode="enc", input_channels=in_ch)
#     convert_to_2d(model.encoder)
#     ckpt = torch.load(pretrained_path, map_location=device)
#     state_dict = ckpt.get("model", ckpt)
#     model.load_state_dict(state_dict, strict=False)
#     model.to(device).eval()
#     return model.encoder



# def get_encoder_features(encoder, x):
#     feats = {}
#     stages = [
#         # ("in_conv", encoder.in_conv),
#         ("encoder_conv1", encoder.encoder_conv1),
#         # ("encoder_conv2", encoder.encoder_conv2),
#         # ("encoder_conv3", encoder.encoder_conv3),
#         # ("encoder_conv4", encoder.encoder_conv4),
#     ]
#     def hook_fn(name):
#         def fn(module, input, output):
#             feats[name] = output
#         return fn
#     handles = []
#     for name, layer in stages:
#         handles.append(layer.register_forward_hook(hook_fn(name)))
#     _ = encoder(x)
#     for h in handles: h.remove()
#     return feats

def edge_loss(pred, target):
    # Sobel edge maps
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)
    sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32, device=pred.device).view(1,1,3,3)

    weight_x = sobel_x.expand(pred.size(1),1,3,3)
    weight_y = sobel_y.expand(pred.size(1),1,3,3)

    edge_pred_x = F.conv2d(pred, weight_x, padding=1, groups=pred.size(1))
    edge_pred_y = F.conv2d(pred, weight_y, padding=1, groups=pred.size(1))
    edge_target_x = F.conv2d(target, weight_x, padding=1, groups=target.size(1))
    edge_target_y = F.conv2d(target, weight_y, padding=1, groups=target.size(1))

    return F.l1_loss(edge_pred_x, edge_target_x) + F.l1_loss(edge_pred_y, edge_target_y)


# def diff_masked_l1(pred, target, eps=1e-6, pct_start=0.7, pct_end=0.4, soft_beta=0.05):
#     """
#     Masked L1 focused on high-error regions using predicted diff_map = |pred - target|.
#     - Compute d = |pred - target|
#     - Construct mask m = sigmoid((d - tau)/beta) where tau is a percentile of d inside batch
#     - Return masked L1 and mask (for monitoring)
#     """
#     d = (pred - target).abs()  # [B,C,H,W]
#     # compute per-sample tau as percentile over spatial+channel dims
#     # flatten per-sample and compute 70th percentile (start)
#     B = d.shape[0]
#     flat = d.view(B, -1)
#     # try/except for older PyTorch versions for torch.quantile
#     try:
#         tau = torch.quantile(flat, pct_start, dim=1).view(B,1,1,1)
#     except Exception:
#         # fallback: median-ish approximation using topk
#         k = max(1, int(flat.shape[1] * pct_start))
#         tau_vals, _ = flat.topk(k, dim=1)
#         tau = tau_vals.min(dim=1)[0].view(B,1,1,1)

#     # beta scaled to range of d (approx)
#     global_max = d.max().detach()
#     beta = (soft_beta * (global_max + eps)).clamp_min(1e-6)

#     m = torch.sigmoid((d - tau) / beta)  # soft mask in (0,1)
#     # normalize mask per sample
#     m_norm = m / (m.mean(dim=[1,2,3], keepdim=True) + 1e-6)

#     masked_l1 = (m * (pred - target).abs()).sum() / (m.sum() + 1e-6)
#     return masked_l1, m

def overlay_saliency_on_pred(pred, saliency):
    """
    pred: [H, W] or [H, W, 3] image (grayscale or RGB)
    saliency: [H, W] saliency normalized to [0, 1]
    Returns: [H, W, 3] RGB image with saliency overlay
    """
    plt.ioff()

    pred = np.squeeze(pred)
    saliency = np.squeeze(saliency)

    # Normalize saliency map
    saliency_norm = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    # Create plot and overlay
    fig, ax = plt.subplots(figsize=(3, 3), dpi=100)
    ax.imshow(pred, cmap='gray')
    ax.imshow(saliency_norm, cmap='jet', alpha=0.5)
    ax.axis("off")
    fig.tight_layout(pad=0)

    # Convert matplotlib figure to image array
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    saliency_overlay = Image.open(buf).convert("RGB")
    saliency_overlay = np.array(saliency_overlay)
    plt.close(fig)

    return saliency_overlay


# def gap_feature_loss(feats_fake, feats_gt, stage_weights=None):
#     loss = 0.0
#     for i, (k, v_fake) in enumerate(feats_fake.items()):
#         v_gt = feats_gt[k].detach()   # <-- detach here instead of at encoder
#         gap_fake = F.adaptive_avg_pool2d(v_fake, 1)  # keeps gradient
#         gap_gt   = F.adaptive_avg_pool2d(v_gt, 1)
#         loss += stage_weights[i] * F.l1_loss(gap_fake, gap_gt)
#     return loss


# def gap_feature_loss_raw(feats_fake, feats_gt, return_per_stage=True):
#     """
#     Compute raw GAP feature loss per stage without any weighting.
#     """
#     total_loss = 0.0
#     per_stage_losses = []

#     for i, (k, v_fake) in enumerate(feats_fake.items()):
#         v_gt = feats_gt[k].detach()  # detach GT to avoid gradient
#         gap_fake = F.adaptive_avg_pool2d(v_fake, 1)
#         gap_gt   = F.adaptive_avg_pool2d(v_gt, 1)
#         loss_stage = F.l1_loss(gap_fake, gap_gt)
#         per_stage_losses.append((k, loss_stage.item()))  # store stage name and loss
#         total_loss += loss_stage

#     if return_per_stage:
#         return total_loss, per_stage_losses
#     return total_loss


# def patch_nce_loss(feat_q, feat_k, temperature=0.07, num_patches=256):
#     """
#     feat_q, feat_k: [B, C, H, W]
#     """
#     B, C, H, W = feat_q.shape
#     feat_k = feat_k.detach()

#     # Flatten features: [B, H*W, C]
#     feat_q_flat = feat_q.permute(0, 2, 3, 1).reshape(B, H*W, C)
#     feat_k_flat = feat_k.permute(0, 2, 3, 1).reshape(B, H*W, C)

#     feat_q_flat = F.normalize(feat_q_flat, dim=-1, p=2)
#     feat_k_flat = F.normalize(feat_k_flat, dim=-1, p=2)


#     # Randomly sample patches if too many
#     if H*W > num_patches:
#         idx = torch.randperm(H*W)[:num_patches]
#         feat_q_flat = feat_q_flat[:, idx, :]  # [B, num_patches, C]
#         feat_k_flat = feat_k_flat[:, idx, :]

#     # Positive logits: [B*num_patches, 1]
#     l_pos = torch.sum(feat_q_flat * feat_k_flat, dim=-1, keepdim=True)
#     l_pos = l_pos.view(-1, 1)  # flatten batch and patches

#     # Negative logits: within sampled patches of same image
#     l_neg = torch.bmm(feat_q_flat, feat_k_flat.transpose(1, 2))  # [B, num_patches, num_patches]
#     # mask diagonal
#     diagonal = torch.eye(l_neg.size(1), device=feat_q.device, dtype=torch.bool)[None, :, :]
#     l_neg.masked_fill_(diagonal, -10.0)
#     l_neg = l_neg.view(-1, l_neg.size(1))

#     # Concatenate pos and neg
#     out = torch.cat([l_pos, l_neg], dim=1) / temperature
#     labels = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
#     loss = F.cross_entropy(out, labels)
#     return loss

# def patch_nce_loss_debug(feat_q, feat_k, temperature=0.07, num_patches=256):
#     """
#     Debug version of PatchNCE loss.
#     feat_q, feat_k: [B, C, H, W]
#     """
#     B, C, H, W = feat_q.shape
#     print(f"\n=== PatchNCE Debug ===")
#     print(f"Input feat_q: {feat_q.shape}, min={feat_q.min().item():.4f}, max={feat_q.max().item():.4f}")
#     print(f"Input feat_k: {feat_k.shape}, min={feat_k.min().item():.4f}, max={feat_k.max().item():.4f}")

#     feat_k = feat_k.detach()

#     # Flatten
#     feat_q_flat = feat_q.permute(0, 2, 3, 1).reshape(B, H*W, C)
#     feat_k_flat = feat_k.permute(0, 2, 3, 1).reshape(B, H*W, C)
#     print(f"Flattened feat_q: {feat_q_flat.shape}, range=({feat_q_flat.min().item():.4f}, {feat_q_flat.max().item():.4f})")
#     print(f"Flattened feat_k: {feat_k_flat.shape}, range=({feat_k_flat.min().item():.4f}, {feat_k_flat.max().item():.4f})")

#     # Sampling
#     if H*W > num_patches:
#         idx = torch.randperm(H*W)[:num_patches]
#         feat_q_flat = feat_q_flat[:, idx, :]
#         feat_k_flat = feat_k_flat[:, idx, :]
#         print(f"Sampled {num_patches} patches: feat_q {feat_q_flat.shape}, feat_k {feat_k_flat.shape}")

#     # Positive logits
#     l_pos = torch.sum(feat_q_flat * feat_k_flat, dim=-1, keepdim=True)
#     l_pos = l_pos.view(-1, 1)
#     print(f"l_pos: {l_pos.shape}, min={l_pos.min().item():.4f}, max={l_pos.max().item():.4f}")

#     # Negative logits
#     l_neg = torch.bmm(feat_q_flat, feat_k_flat.transpose(1, 2))
#     diagonal = torch.eye(l_neg.size(1), device=feat_q.device, dtype=torch.bool)[None, :, :]
#     l_neg.masked_fill_(diagonal, -10.0)
#     l_neg = l_neg.view(-1, l_neg.size(1))
#     print(f"l_neg: {l_neg.shape}, min={l_neg.min().item():.4f}, max={l_neg.max().item():.4f}")

#     # Combine
#     out = torch.cat([l_pos, l_neg], dim=1) / temperature
#     print(f"out logits: {out.shape}, min={out.min().item():.4f}, max={out.max().item():.4f}")

#     labels = torch.zeros(out.size(0), dtype=torch.long, device=feat_q.device)
#     print(f"labels: {labels.shape}, unique={labels.unique()}")

#     loss = F.cross_entropy(out, labels)
#     print(f"loss: {loss.item():.4f}")
#     print("=========================\n")

#     return loss





class ResViT_model(BaseModel):
    def name(self):
        return 'ResViT_model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.psnr_values = []  # Store PSNR for all images
        self.ssim_values = []  # Store SSIM for all images
        
        # Initialize in your model's __init__ or setup
        self.ema_diffmap = None
        self.ema_edge = None
        self.ema_fft = None
        self.ema_decay = 0.99  # smoothing factor

        # load/define networks
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # pretrained_path = "/home/fnu.talha/resvit/unet_b.ckpt"
        # pretrained_encoder = load_encoder(pretrained_path, in_ch=3, device=device)
        # self.encoder = load_encoder(pretrained_path, in_ch=1, device=device)
        # self.seg_net = load_segmentation_unet(pretrained_path, in_ch=1, device=device)

        # Initialize SSIM metric once (e.g., in __init__ of your model)
        self.ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        # my changes here
        self.netG = networks.define_G(3, opt.output_nc, opt.ngf,
                                      opt.which_model_netG,opt.vit_name,opt.fineSize,opt.pre_trained_path, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids,
                                      pre_trained_trans=opt.pre_trained_transformer,pre_trained_resnet = opt.pre_trained_resnet)
        self.edge_feats = {}
        # Initialize attributes
        self.saliency = None  # Ensure this attribute always exists

        if self.isTrain:
            self.lambda_f = opt.lambda_f
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.input_nc + opt.output_nc-1, opt.ndf,
                                          opt.which_model_netD,opt.vit_name,opt.fineSize,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

        def hook_edge_proj_1(module, input, output):
            # output is edge_proj(edge): [B,C,H,W]
            self.edge_feats['encoder1_edge'] = output.detach()

        # self.netG.encoder_1[4].edge_proj.register_forward_hook(hook_edge_proj_1)

        # # similarly you can hook encoder_2 and encoder_3 if you want:
        # def hook_edge_proj_2(module, input, output):
        #     self.edge_feats['encoder2_edge'] = output.detach()
        # # self.netG.encoder_2[3].edge_proj.register_forward_hook(hook_edge_proj_2)

        # def hook_edge_proj_3(module, input, output):
        #     self.edge_feats['encoder3_edge'] = output.detach()
        # # self.netG.encoder_3[3].edge_proj.register_forward_hook(hook_edge_proj_3)

        # self.encoder_feats = {}

        # def hook_enc1(module, inp, out):
        #     self.encoder_feats['encoder_1'] = out.detach()

        # def hook_enc2(module, inp, out):
        #     self.encoder_feats['encoder_2'] = out.detach()

        # def hook_enc3(module, inp, out):
        #     self.encoder_feats['encoder_3'] = out.detach()

        # self.netG.encoder_1[-1].register_forward_hook(hook_enc1)
        # self.netG.encoder_2[-1].register_forward_hook(hook_enc2)
        # self.netG.encoder_3[-1].register_forward_hook(hook_enc3)

    # def get_encoder_heatmap(self, enc_key='enc1'):
    #     """
    #     Convert stored encoder feature map to [H,W] heatmap (mean |activation| over channels).
    #     enc_key: 'enc1', 'enc2', 'enc3'
    #     """
    #     if enc_key not in self.encoder_feats:
    #         raise ValueError(f"No stored encoder features for {enc_key}. "
    #                          "Did you run model.test() first?")

    #     feat = self.encoder_feats[enc_key]   # [B,C,H,W]
    #     heat = feat.abs().mean(dim=1)        # [B,H,W]
    #     heat = heat[0].cpu().numpy()         # [H,W]

    #     hmin, hmax = heat.min(), heat.max()
    #     if hmax > hmin:
    #         heat = (heat - hmin) / (hmax - hmin)
    #     else:
    #         heat = np.zeros_like(heat)
    #     return heat


    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], non_blocking=True)
            input_B = input_B.cuda(self.gpu_ids[0], non_blocking=True)
        self.input_A = input_A
        self.input_B = input_B
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    # def get_rdeb_heatmap(self, layer='encoder_1'):
    #     """
    #     Extract mean-absolute heatmap from a given RDEB layer.
    #     Assumes self.real_A is already set (shape [B,3,H,W]: T1,T2,FLAIR).
    #     layer: 'encoder_1', 'encoder_2', 'encoder_3'
    #     Returns: heatmap [H,W] normalized to [0,1]
    #     """
    #     with torch.no_grad():
    #         x = self.real_A[:, 0:3, :, :]  # use 3-channel RGB-style input

    #         if layer == 'encoder_1':
    #             # encoder_1 = [pad, conv7x7, BN, ReLU, RDEB]
    #             x = self.netG.encoder_1[0:4](x)
    #             rdeb_out = self.netG.encoder_1[4](x)

    #         elif layer == 'encoder_2':
    #             # through encoder_1
    #             x = self.netG.encoder_1(x)
    #             # encoder_2 = [conv, BN, ReLU, RDEB]
    #             x = self.netG.encoder_2[0:3](x)
    #             rdeb_out = self.netG.encoder_2[3](x)

    #         elif layer == 'encoder_3':
    #             x = self.netG.encoder_1(x)
    #             x = self.netG.encoder_2(x)
    #             x = self.netG.encoder_3[0:3](x)
    #             rdeb_out = self.netG.encoder_3[3](x)

    #         else:
    #             raise ValueError(f'Invalid layer name for RDEB: {layer}')

    #         # rdeb_out: [B, C, H, W] → mean over channels
    #         heatmap = rdeb_out.abs().mean(dim=1).cpu().numpy()[0]  # [H,W]
    #         hmin, hmax = heatmap.min(), heatmap.max()
    #         if hmax > hmin:
    #             heatmap = (heatmap - hmin) / (hmax - hmin)
    #         else:
    #             heatmap = np.zeros_like(heatmap)

    #         return heatmap

    def compute_saliency_map(self, use_gt=True):
        """
        Compute gradient-based saliency map w.r.t. 3-channel input (T1,T2,FLAIR).
        If use_gt=True, uses self.real_B as GT; otherwise uses fake_B as pseudo-GT.
        Returns: saliency [H,W] normalized to [0,1]
        """
        # clone input, enable grad
        input_img = self.input_A[:, 0:3, :, :].clone().detach().requires_grad_(True)
        self.netG.eval()

        with torch.enable_grad():
            fake_B = self.netG(input_img)
            if use_gt and hasattr(self, 'input_B'):
                target = self.input_B
            else:
                target = fake_B.detach()

            loss = F.mse_loss(fake_B, target)
            grads = torch.autograd.grad(
                outputs=loss,
                inputs=input_img,
                create_graph=False,
                retain_graph=False
            )[0]  # [B,3,H,W]

        saliency = grads.abs().mean(dim=1).detach().cpu().numpy()[0]  # [H,W]
        smin, smax = saliency.min(), saliency.max()
        if smax > smin:
            saliency = (saliency - smin) / (smax - smin)
        else:
            saliency = np.zeros_like(saliency)

        return saliency


    # def get_edgeproj_heatmap(self, encoder='encoder1_edge'):
    #     """
    #     Turn edge_proj feature map from a chosen encoder RDEB into a [H,W] heatmap.
    #     encoder: 'encoder1_edge', 'encoder2_edge', 'encoder3_edge'
    #     """
    #     assert encoder in self.edge_feats, f"No stored edge features for {encoder}"

    #     feat = self.edge_feats[encoder]  # [B,C,H,W]
    #     # mean-abs over channels
    #     heatmap = feat.abs().mean(dim=1).cpu().numpy()[0]  # [H,W]

    #     hmin, hmax = heatmap.min(), heatmap.max()
    #     if hmax > hmin:
    #         heatmap = (heatmap - hmin) / (hmax - hmin)
    #     else:
    #         heatmap = np.zeros_like(heatmap)
    #     return heatmap

    def forward(self, compute_saliency=True):
        # Forward pass through the generator
        self.real_A = Variable(self.input_A)
        # my changes here
        self.fake_B = self.netG(self.real_A[:, 0:3, :, :])  # Generate fake B from real A
        # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])

        self.real_B = Variable(self.input_B)  # Ground truth B (real B)

        # If we need to compute saliency, do it here
        if compute_saliency:
            # Enable gradient tracking for saliency computation
            self.real_A.requires_grad = True  # Ensure gradients can be computed
            
            # Recompute the fake B with gradient tracking enabled
            self.fake_B = self.netG(self.real_A[:, 0:3, :, :])
            # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])

            # Compute MSE loss for saliency
            loss = torch.nn.functional.mse_loss(self.fake_B, self.real_B)

            # Compute gradients of the loss with respect to the input (real_A)
            gradients = torch.autograd.grad(outputs=loss, inputs=self.real_A, create_graph=False, retain_graph=True)[0]

            # Compute saliency: absolute gradients for each pixel in the image
            saliency = gradients.abs().mean(dim=1).cpu().numpy().squeeze()  # Mean across channels

            if saliency.max() > 0 and saliency.max() != saliency.min():
                self.saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())
            else:
                self.saliency = np.zeros_like(saliency)


    def test(self, compute_saliency=False):
        self.saliency = None  # Reset saliency before computation
        torch.cuda.empty_cache()

        if compute_saliency:
            # Enable gradient tracking
            self.real_A = self.input_A.clone().detach().requires_grad_(True)
            self.real_B = self.input_B
            self.fake_B = self.netG(self.real_A[:, 0:3, :, :])
            # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])

            loss = torch.nn.functional.mse_loss(self.fake_B, self.real_B)
            gradients = torch.autograd.grad(outputs=loss, inputs=self.real_A, create_graph=False, retain_graph=True)[0]
            saliency = gradients.abs().mean(dim=1).cpu().numpy().squeeze()

            if saliency.max() > 0:
                self.saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min())

        else:
            with torch.no_grad():
                self.real_A = self.input_A
                self.fake_B = self.netG(self.real_A[:, 0:3, :, :])
                # self.fake_B, self.edge_pred, self.diff_pred = self.netG(self.real_A[:, 0:3, :, :])
                self.real_B = self.input_B

        # Detach tensors and convert to numpy for visualization
        fake_B_np = self.fake_B.detach().cpu().numpy().squeeze()
        real_B_np = self.real_B.detach().cpu().numpy().squeeze()

        # Normalize from [-1, 1] range to [0, 1]
        fake_B_np = (fake_B_np + 1) / 2
        real_B_np = (real_B_np + 1) / 2

        # Clip to ensure values are strictly in [0, 1]
        fake_B_np = np.clip(fake_B_np, 0, 1)
        real_B_np = np.clip(real_B_np, 0, 1)

        # Handle zero-max cases before PSNR computation
        if real_B_np.max() <= 0 or fake_B_np.max() <= 0:
            return  # Skip this image if there's no valid data

        # Compute PSNR and SSIM
        psnr_value = psnr(real_B_np, fake_B_np, data_range=1.0)
        ssim_value = ssim(real_B_np, fake_B_np, data_range=1.0, multichannel=True)

        # Store values for averaging
        self.psnr_values.append(psnr_value)
        self.ssim_values.append(ssim_value)

        # Compute Difference Map
        diff_map = np.abs(fake_B_np - real_B_np)
        # Normalize difference map to [0, 1] for visualization
        diff_map = np.clip(diff_map / diff_map.max(), 0, 1)

        print(f"Image {len(self.psnr_values)} | PSNR: {psnr_value}, SSIM: {ssim_value}")

        # Return necessary data for visualization and further processing
        return fake_B_np, real_B_np, diff_map, self.saliency


    def compute_final_metrics(self):
        """Compute and print average and standard deviation for PSNR & SSIM after all images are processed."""
        if self.psnr_values and self.ssim_values:
            # Convert lists to PyTorch tensors
            psnr_tensor = torch.tensor(self.psnr_values, dtype=torch.float32)
            ssim_tensor = torch.tensor(self.ssim_values, dtype=torch.float32)

            # Create masks for finite values (excludes `inf` and `NaN`)
            psnr_finite_mask = torch.isfinite(psnr_tensor)
            ssim_finite_mask = torch.isfinite(ssim_tensor)

            # Filter out `inf` and `NaN` values using the masks
            psnr_filtered = psnr_tensor[psnr_finite_mask]
            ssim_filtered = ssim_tensor[ssim_finite_mask]

            # Compute mean and std for PSNR (only on finite values)
            avg_psnr = torch.mean(psnr_filtered) if psnr_filtered.numel() > 0 else torch.tensor(float('nan'))
            std_psnr = torch.std(psnr_filtered) if psnr_filtered.numel() > 0 else torch.tensor(float('nan'))

            # Compute mean and std for SSIM (only on finite values)
            avg_ssim = torch.mean(ssim_filtered) if ssim_filtered.numel() > 0 else torch.tensor(float('nan'))
            std_ssim = torch.std(ssim_filtered) if ssim_filtered.numel() > 0 else torch.tensor(float('nan'))

            print(f"\nFinal Average PSNR: {avg_psnr.item():.3f} ± {std_psnr.item():.3f}, Final Average SSIM: {avg_ssim.item():.3f} ± {std_ssim.item():.3f}\n")

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_AB = self.fake_AB_pool.query(torch.cat((self.real_A[:,0:3,:,:], self.fake_B), 1).data)
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) #
        # Real
        real_AB = torch.cat((self.real_A[:,0:3,:,:], self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5*self.opt.lambda_adv

        self.loss_D.backward()


    def update_ema(self, ema, value):
        if ema is None:
            return value.detach()
        return self.ema_decay * ema + (1 - self.ema_decay) * value.detach()


    # def backward_G(self):
    #     # --- GAN loss ---
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)

    #     # --- L1 loss ---
    #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10

    #     # --- Diff_map loss ---
    #     if self.opt.lambda_diffmap > 0:
    #         diff_map = torch.abs(self.real_B - self.fake_B)
    #         self.loss_diffmap = torch.mean(diff_map)

    #         # Update EMA
    #         self.ema_diffmap = self.update_ema(self.ema_diffmap, self.loss_diffmap)

    #         # Adaptive scaling around initial λ=10
    #         scale = self.opt.lambda_diffmap * (self.loss_diffmap.item() / (self.ema_diffmap.item() + 1e-8))
    #         self.loss_G_diffmap = scale * self.loss_diffmap
    #     else:
    #         self.loss_G_diffmap = 0

    #     # --- Edge loss ---
    #     if getattr(self.opt, "lambda_edge", 0) > 0:
    #         raw_edge_loss = edge_loss(self.fake_B, self.real_B)
    #         self.ema_edge = self.update_ema(self.ema_edge, raw_edge_loss)

    #         scale = self.opt.lambda_edge * (raw_edge_loss.item() / (self.ema_edge.item() + 1e-8))
    #         self.loss_G_edge = scale * raw_edge_loss
    #     else:
    #         self.loss_G_edge = torch.tensor(0.0, device=self.fake_B.device)

    #     # --- Total loss ---
    #     self.loss_G = (
    #         self.loss_G_GAN
    #         + self.loss_G_L1
    #         + self.loss_G_diffmap
    #         + self.loss_G_edge
    #     )

    #     self.loss_G.backward()


    # def backward_G(self):
    #     """
    #     Compute generator losses and backpropagate.
    #     Uses stored outputs from forward pass:
    #     - self.fake_B : generated image
    #     - self.edge_pred : predicted edge map (optional)
    #     - self.diff_pred : predicted difference map (optional)
    #     """
    #     fake_B = self.fake_B
    #     edge_pred = getattr(self, "edge_pred", None)
    #     diff_pred = getattr(self, "diff_pred", None)
    #     # print("Generator outputs:")
    #     # print("fake_B:", fake_B.shape, fake_B.min().item(), fake_B.max().item())
    #     # print("edge_pred:", edge_pred.shape, edge_pred.min().item(), edge_pred.max().item())
    #     # if diff_pred is not None:
    #     #     print("diff_pred:", diff_pred.shape, diff_pred.min().item(), diff_pred.max().item())


    #     # --- Adversarial loss ---
    #     fake_AB = torch.cat((self.real_A, fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
    #     # print("G_GAN:", self.loss_G_GAN.item())

    #     # --- Pixel-wise L1 loss ---
    #     self.loss_G_L1 = self.criterionL1(fake_B, self.real_B) * 5

    #     # --- Self-supervised diff map loss (optional) ---
    #     if getattr(self.opt, "lambda_diffmap", 0) > 0 and diff_pred is not None:
    #         self.loss_diffmap = torch.mean(diff_pred)
    #         self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
    #         # print("G_diff_map:", self.loss_G_diffmap.item(), "range:", self.diff_map.min().item(), self.diff_map.max().item())

    #     else:
    #         self.loss_G_diffmap = torch.tensor(0.0, device=fake_B.device)

            

    #     # --- Edge consistency loss (optional) ---
    #     if getattr(self.opt, "lambda_edge", 0) > 0 and edge_pred is not None:
    #         self.loss_G_edge = edge_loss(edge_pred, self.real_B)
    #         self.loss_G_edge = self.opt.lambda_edge * self.loss_G_edge
    #         # print("G_edge_loss:", self.loss_G_edge.item())

    #     else:
    #         self.loss_G_edge = torch.tensor(0.0, device=fake_B.device)

    #     # --- Total Generator loss ---
    #     self.loss_G = (
    #         self.loss_G_GAN
    #         + self.loss_G_L1
    #         + self.loss_G_diffmap
    #         + self.loss_G_edge
    #     )

    #     # Backprop
    #     self.loss_G.backward()

    def backward_G(self):
        # --- Adversarial loss ---
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # --- Pixel-wise L1 loss ---
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
        self.loss_G_L1 = self.loss_G_L1 * 10


        if self.opt.lambda_diffmap > 0:
            # Allow gradient flow through fake_B
            diff_map = torch.abs(self.real_B - self.fake_B)

            # Normalize the difference map (optional)
            # diff_map = diff_map / (diff_map.max() + 1e-8)

            # Compute the auxiliary loss
            self.loss_diffmap = torch.mean(diff_map)
            self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
        else:
            self.loss_G_diffmap = 0

        # if self.opt.lambda_diffmap > 0:
        #     # Pixel-wise absolute difference
        #     diff_map = torch.abs(self.real_B - self.fake_B)    # [B, C, H, W]

        #     # Step 1: GAP per image (mean over pixels/channels)
        #     per_image_loss = diff_map.view(diff_map.size(0), -1).mean(dim=1)  # [B]

        #     # Step 2: Mean over the batch
        #     self.loss_diffmap = per_image_loss.mean()

        #     # Scale with lambda
        #     self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
        # else:
        #     self.loss_G_diffmap = 0



        # --- Self-supervised loss: difference map ---
        # if self.opt.lambda_diffmap > 0:
        #     diff_map = torch.abs(self.real_B - self.fake_B)

        #     # --- Soft diff_map weighting in [0.1,1.0] ---
        #     eps = 1e-8
        #     w = torch.sigmoid(
        #         (diff_map - diff_map.mean(dim=[2,3], keepdim=True)) / 
        #         (diff_map.std(dim=[2,3], keepdim=True) + eps)
        #     )
        #     w = 0.1 + 0.9 * (w - w.min()) / (w.max() - w.min() + eps)

        #     self.loss_diffmap = torch.mean(diff_map)
        #     self.loss_G_diffmap = self.opt.lambda_diffmap * torch.mean(w * diff_map)
        # else:
        #     self.loss_G_diffmap = 0
        

        # --- Edge loss ---
        if getattr(self.opt, "lambda_edge", 0) > 0:
            self.loss_G_edge = edge_loss(self.fake_B, self.real_B)
            self.loss_G_edge = self.opt.lambda_edge * self.loss_G_edge
        else:
            self.loss_G_edge = torch.tensor(0.0, device=self.fake_B.device)



        # # --- FFT high-pass loss ---
        # if getattr(self.opt, "lambda_fft", 0) > 0:
        #     self.loss_G_fft = self.fft_highpass_loss(self.fake_B, self.real_B, radius_frac=0.06)
        #     # Optional: multiply by soft diff_map to focus on high-error regions
        #     self.loss_G_fft = self.opt.lambda_fft * self.loss_G_fft
        # else:
        #     self.loss_G_fft = torch.tensor(0.0, device=self.fake_B.device)
        
        # --- FFT high-pass loss with EMA-based stabilization ---
        if getattr(self.opt, "lambda_fft", 0) > 0:
            # 1 Compute raw FFT loss
            raw_loss_G_fft = self.fft_highpass_loss(self.fake_B, self.real_B, radius_frac=0.06)

            # 2 Initialize or update EMA (detached!)
            if self.ema_fft is None:
                self.ema_fft = raw_loss_G_fft.detach()
            else:
                self.ema_fft = self.ema_decay * self.ema_fft + (1 - self.ema_decay) * raw_loss_G_fft.detach()

            # 3 Compute a soft-normalized loss for stability
            # Scale by EMA to reduce batch-to-batch std but preserve magnitude
            normalized_fft = raw_loss_G_fft * (self.ema_fft.detach() / (raw_loss_G_fft.detach() + 1e-6))

            # 4 Clamp to avoid runaway spikes (optional)
            normalized_fft = torch.clamp(normalized_fft, 0.5 * raw_loss_G_fft, 2.0 * raw_loss_G_fft)

            # 5 Final FFT loss
            self.loss_G_fft = self.opt.lambda_fft * normalized_fft

        else:
            self.loss_G_fft = torch.tensor(0.0, device=self.fake_B.device)


        # # --- FFT loss ---
        # if getattr(self.opt, "lambda_fft", 0) > 0:
        #     raw_fft_loss = self.fft_highpass_loss(self.fake_B, self.real_B)

        #     # Optional warmup (recommended)
        #     if hasattr(self, "current_epoch"):
        #         progress = min(1.0, self.current_epoch / 30.0)  # 30-epoch warmup
        #     else:
        #         progress = 1.0

        #     lambda_fft_scaled = self.opt.lambda_fft * progress

        #     self.loss_G_fft = lambda_fft_scaled * raw_fft_loss
        # else:
        #     self.loss_G_fft = torch.tensor(0.0, device=self.fake_B.device)



        # --- SSIM loss (torchmetrics) ---
        # --- SSIM loss with proper normalization ---
        # if getattr(self.opt, "lambda_ssim", 0) > 0:
        #     # Normalize to [0,1] for SSIM
        #     fake_norm = (self.fake_B + 1.0) / 2.0
        #     real_norm = (self.real_B + 1.0) / 2.0
        #     self.loss_G_ssim = self.opt.lambda_ssim * (1 - self.ssim_metric(fake_norm, real_norm))
        # else:
        #     self.loss_G_ssim = torch.tensor(0.0, device=self.fake_B.device)

        # inside backward_G
        # --- Auxiliary multi-scale L1 loss ---
        # if getattr(self.opt, "lambda_ms", 0) > 0:
        #     import torch.nn.functional as F

        #     ms_loss = 0
        #     scales = [0.5, 0.25]     # only coarse scales
        #     weights = [0.5, 0.25]    # smaller contribution

        #     for s, w in zip(scales, weights):
        #         H, W = int(self.real_B.size(2) * s), int(self.real_B.size(3) * s)
        #         fake_s = F.interpolate(self.fake_B, size=(H, W), mode='bilinear', align_corners=False)
        #         real_s = F.interpolate(self.real_B, size=(H, W), mode='bilinear', align_corners=False)
        #         ms_loss += w * F.l1_loss(fake_s, real_s)

        #     self.loss_G_ms = self.opt.lambda_ms * ms_loss
        # else:
        #     self.loss_G_ms = torch.tensor(0.0, device=self.fake_B.device)


        # --- Combine all losses ---
        self.loss_G = (
            self.loss_G_GAN
            + self.loss_G_L1
            + self.loss_G_diffmap
            + self.loss_G_edge
            + self.loss_G_fft
        )

        self.loss_G.backward()


    # --- Pixel-wise L1 loss (multi-scale) ---
    # def l1_multiscale(fake, real, scales=[1.0, 0.5, 0.25], weights=[1.0, 0.5, 0.25]):
    #     import torch.nn.functional as F
    #     total_loss = 0
    #     for s, w in zip(scales, weights):
    #         if s != 1.0:
    #             H, W = int(real.size(2) * s), int(real.size(3) * s)
    #             fake_s = F.interpolate(fake, size=(H, W), mode='bilinear', align_corners=False)
    #             real_s = F.interpolate(real, size=(H, W), mode='bilinear', align_corners=False)
    #         else:
    #             fake_s, real_s = fake, real
    #         total_loss += w * F.l1_loss(fake_s, real_s)
    #     return total_loss



    # --- FFT high-pass loss function ---
    def fft_highpass_loss(self, pred, target, radius_frac=0.08):
        B,C,H,W = pred.shape
        Fp = torch.fft.fftshift(torch.fft.fft2(pred.squeeze(1)), dim=(-2,-1))
        Ft = torch.fft.fftshift(torch.fft.fft2(target.squeeze(1)), dim=(-2,-1))
        yy, xx = torch.meshgrid(torch.arange(H, device=pred.device), torch.arange(W, device=pred.device))
        cy, cx = H//2, W//2
        r = torch.sqrt((yy-cy).float()**2 + (xx-cx).float()**2)
        r = r / r.max()
        mask = (r > radius_frac).float()

        mag_diff = torch.abs(Fp - Ft)
        masked = mag_diff * mask

        # print("FFT masked mean:", masked.mean().item())  # debug
        return masked.mean()

    # def fft_highpass_loss(self, pred, target, radius_frac=0.08, eps=1e-6):
    #     """
    #     Stable yet effective FFT high-frequency loss.
    #     """
    #     B, C, H, W = pred.shape

    #     # FFT and magnitude
    #     Fp = torch.fft.fftshift(torch.fft.fft2(pred.squeeze(1)), dim=(-2, -1))
    #     Ft = torch.fft.fftshift(torch.fft.fft2(target.squeeze(1)), dim=(-2, -1))
    #     Mp = torch.abs(Fp)
    #     Mt = torch.abs(Ft)

    #     # Radial mask (soft + minimal threshold for stability)
    #     yy, xx = torch.meshgrid(
    #         torch.arange(H, device=pred.device),
    #         torch.arange(W, device=pred.device),
    #         indexing="ij"
    #     )
    #     cy, cx = H // 2, W // 2
    #     r = torch.sqrt((yy - cy).float()**2 + (xx - cx).float()**2)
    #     r = r / r.max()

    #     weight = torch.clamp((r - radius_frac) / (1 - radius_frac), 0, 1)

    #     # Relative difference, but scale to match old loss magnitude
    #     rel_diff = torch.abs(Mp - Mt) / (Mt + eps)
    #     loss = (rel_diff * weight).mean() * 0.4  # scale factor ~0.4 to keep magnitude similar to old loss

    #     return loss




    # def backward_G(self):
    #     # --- Adversarial loss ---
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)

    #     # --- Pixel-wise L1 loss ---
    #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)
    #     self.loss_G_L1  = self.loss_G_L1*10
    #     # --- Self-supervised loss: difference map ---
    #     if self.opt.lambda_diffmap > 0:
    #         # Allow gradient flow through fake_B
    #         diff_map = torch.abs(self.real_B - self.fake_B)

    #         # Normalize the difference map (optional)
    #         # diff_map = diff_map / (diff_map.max() + 1e-8)

    #         # Compute the auxiliary loss
    #         self.loss_diffmap = torch.mean(diff_map)
    #         self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
    #     else:
    #         self.loss_G_diffmap = 0

        # print(self.real_B.shape)
        # print(self.fake_B.shape)

    # # --- Multi-layer FOMO perceptual loss ---
    #     if self.opt.lambda_percep > 0:
    #         with torch.no_grad():
    #             feats_gt = get_encoder_features(self.encoder, self.real_B)

    #         # keep grad for fake_B path
    #         feats_fake = get_encoder_features(self.encoder, self.fake_B)

    #         def normalize_per_channel(x, eps=1e-6):
    #             B, C, H, W = x.shape
    #             mean = x.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    #             std  = x.view(B, C, -1).std(dim=2).view(B, C, 1, 1) + eps
    #             return (x - mean) / std

    #         perceptual_loss = 0.0
    #         for name in feats_fake.keys():
    #             f_fake = normalize_per_channel(feats_fake[name])
    #             f_gt   = normalize_per_channel(feats_gt[name])
    #             perceptual_loss += F.l1_loss(f_fake, f_gt)
    #         self.loss_G_percep = self.opt.lambda_percep * perceptual_loss
    #     else:
    #         self.loss_G_percep = 0

        # perceptual_loss = F.l1_loss(feats_fake, feats_gt)


        # import matplotlib.pyplot as plt

        # def visualize_encoder_features(feats_gt, feats_fake, max_channels=4, save_dir="debug_features", step=0):
        #     import os
        #     os.makedirs(save_dir, exist_ok=True)

        #     for stage, (f_gt, f_fake) in enumerate(zip(feats_gt.values(), feats_fake.values())):
        #         num_ch = min(max_channels, f_gt.shape[1])  # first few channels only
        #         for ch in range(num_ch):
        #             plt.figure(figsize=(8, 4))
                    
        #             # Ground truth feature
        #             plt.subplot(1, 2, 1)
        #             plt.imshow(f_gt[0, ch].detach().cpu(), cmap="viridis")
        #             plt.title(f"GT Stage {stage} Ch {ch}")
        #             plt.axis("off")
                    
        #             # Fake feature
        #             plt.subplot(1, 2, 2)
        #             plt.imshow(f_fake[0, ch].detach().cpu(), cmap="viridis")
        #             plt.title(f"Fake Stage {stage} Ch {ch}")
        #             plt.axis("off")
                    
        #             plt.tight_layout()
        #             plt.savefig(os.path.join(save_dir, f"step{step}_stage{stage}_ch{ch}.png"))
        #             plt.close()

        # visualize_encoder_features(feats_gt, feats_fake, max_channels=4, save_dir="debug_features", step=1)

        # # Feature loss (GAP pooled features)
        # self.loss_feat = gap_feature_loss(feats_fake, feats_gt, stage_weights = [0.15, 0.25, 0.25, 0.25])  # conv_1 → conv4)
        # self.loss_feat, per_stage_losses = gap_feature_loss_raw(feats_fake, feats_gt)

    

        # --- Total Generator loss ---
        # self.loss_G_focus =self.loss_G_focus*50
        # Extract features (already implemented)
        # self.loss_patchnce1 = patch_nce_loss(feats_fake['encoder_conv1'], feats_gt['encoder_conv1'], num_patches=256)
        # self.loss_patchnce2 = patch_nce_loss(feats_fake['encoder_conv2'], feats_gt['encoder_conv2'], num_patches=256)/256

        # self.loss_patchnce1 = 0.5*self.loss_patchnce1
        # self.loss_patchnce2 = 0.5*self.loss_patchnce2

        # self.loss_patchnce = self.loss_patchnce1 + self.loss_patchnce2
        # self.loss_patchnce = self.loss_patchnce*0.1

        # PatchNCE loss
        # self.loss_patchnce = patchnce_loss_fn(feats_fake, feats_gt)

        # if getattr(self.opt, "lambda_edge", 0) > 0:
        #     self.loss_G_edge = edge_loss(self.fake_B, self.real_B)
        #     self.loss_G_edge= self.opt.lambda_edge * self.loss_G_edge
        # else:
        #     self.loss_G_edge = torch.tensor(0.0, device=self.fake_B.device)


        # # --- AFP Loss (Anatomical Feature-Prioritized) ---
        # if self.opt.lambda_afp > 0:
        #     with torch.no_grad():
        #         feats_gt = get_segmentation_features(self.seg_net, self.real_B)
        #     feats_fake = get_segmentation_features(self.seg_net, self.fake_B)

 
        #     self.loss_afp_x1 = self.criterionL1(normalize_features(feats_fake['x1']), normalize_features(feats_gt['x1']))
        #     self.loss_afp_x2 = self.criterionL1(normalize_features(feats_fake['x2']), normalize_features(feats_gt['x2']))

        #     self.loss_G_afp = self.opt.lambda_afp*(self.loss_afp_x1 + self.loss_afp_x2)/2

        #     # Component losses
        #     self.loss_G_afp_x1 = torch.mean(torch.abs(feats_fake["x1"] - feats_gt["x1"]))
        #     self.loss_G_afp_logits = torch.mean(torch.abs(feats_fake["logits"] - feats_gt["logits"]))

        #     # Weighted total AFP loss
        #     afp_loss = 0.7 * self.loss_G_afp_x1 + 0.3 * self.loss_G_afp_logits
        #     self.loss_G_afp = self.opt.lambda_afp * afp_loss
        # else:
        #     self.loss_G_afp_x1 = 0
        #     self.loss_G_afp_logits = 0
        #     self.loss_G_afp = 0

        # --- Total Generator loss ---
        # self.loss_G = (
        #     self.loss_G_GAN
        #     + self.loss_G_L1
        #     + self.loss_G_diffmap
        #     + self.loss_G_edge   # your edge consistency loss
        # )

        # self.loss_G.backward()


        # self.loss_G = self.loss_G_GAN + self.loss_G_L1*5 + self.loss_G_diffmap + 10.0 * self.loss_feat #+ self.loss_G_focus
        # self.loss_G.backward()

    # def update_loss_weights(self, current_losses, beta=0.9):
    #     """
    #     Update adaptive weights based on current loss magnitudes.
    #     Uses EMA smoothing to prevent oscillations.
        
    #     Args:
    #         current_losses: dict of current scalar loss values
    #         beta: EMA decay factor
    #     """
    #     if not hasattr(self, "loss_ema"):
    #         # Initialize EMA dict on first call
    #         self.loss_ema = {k: v for k, v in current_losses.items()}

    #     # Update EMA
    #     for k, v in current_losses.items():
    #         self.loss_ema[k] = beta * self.loss_ema[k] + (1 - beta) * v

    #     # Compute inverse magnitudes → normalize to mean 1
    #     inv_mags = {k: 1.0 / (v + 1e-8) for k, v in self.loss_ema.items()}
    #     mean_inv = sum(inv_mags.values()) / len(inv_mags)

    #     # Normalize weights so average weight = 1
    #     self.loss_weights = {k: (w / mean_inv) for k, w in inv_mags.items()}


    # def backward_G(self):
    #     # --- Adversarial loss ---
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)

    #     # --- Pixel-wise L1 loss ---
    #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

    #     # --- Self-supervised loss: difference map ---
    #     if self.opt.lambda_diffmap > 0:
    #         diff_map = torch.abs(self.real_B - self.fake_B)
    #         self.loss_diffmap = torch.mean(diff_map)
    #         self.loss_G_diffmap = self.opt.lambda_diffmap * self.loss_diffmap
    #     else:
    #         self.loss_G_diffmap = 0.0

    #     # --- Encoder feature losses ---
    #     with torch.no_grad():
    #         feats_gt = get_encoder_features(self.encoder, self.real_B)
    #     feats_fake = get_encoder_features(self.encoder, self.fake_B)

    #     self.loss_patchnce1 = patch_nce_loss(
    #         feats_fake['encoder_conv1'],
    #         feats_gt['encoder_conv1'],
    #         num_patches=256
    #     )

    #     # --- Update adaptive weights ---
    #     current_losses = {
    #         "GAN": float(self.loss_G_GAN.detach()),
    #         "L1": float(self.loss_G_L1.detach()),
    #         "diffmap": float(self.loss_G_diffmap.detach()),
    #         "patchnce": float(self.loss_patchnce1.detach())
    #     }
    #     self.update_loss_weights(current_losses)

    #     # --- Total Generator loss with adaptive weights ---
    #     self.loss_G = (
    #         self.loss_weights["GAN"]     * self.loss_G_GAN +
    #         self.loss_weights["L1"]      * self.loss_G_L1 +
    #         self.loss_weights["diffmap"] * self.loss_G_diffmap +
    #         self.loss_weights["patchnce"]* self.loss_patchnce1
    #     )

    #     self.loss_G.backward()



    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                            ('G_L1', self.loss_G_L1.item()),
                            ('G_diff_map', self.loss_G_diffmap.item()),
                            ('G_FFT', self.loss_G_fft.item()),
                            # ('afp_enc2', self.loss_afp_x2.item()),
                            # ('afp', self.loss_G_afp.item()),
                            ('G_edge_loss', self.loss_G_edge.item()),
                            ('D_real', self.loss_D_real.item()),
                            ('D_fake', self.loss_D_fake.item())
                            ])



    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)

        # Extract T1 and T2
        T1 = real_A[:, :, 0]
        T2 = real_A[:, :, 1]
        Flair = real_A[:, :, 2]

        real_T1 = np.stack([T1, T1, T1], axis=-1)
        real_T2 = np.stack([T2, T2, T2], axis=-1)
        real_Flair = np.stack([Flair,Flair,Flair], axis=-1)

        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)

        diff_map = np.abs(fake_B.astype(np.float32) - real_B.astype(np.float32))
        diff_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8) * 255
        diff_map = diff_map.astype(np.uint8)
        
        # Select the first image in the batch for the diff_map
        diff_map = torch.abs(self.real_B[0] - self.fake_B[0]).detach().cpu().numpy().squeeze()  # Selecting first image
        diff_map = diff_map / (diff_map.max() + 1e-8)  # Normalize between 0 and 1
        diff_map = (diff_map * 255).astype(np.uint8)  # Scale to 0-255 range for visualization
            # Convert diff_map to RGB by stacking it as 3 channels
        diff_map = np.stack([diff_map] * 3, axis=-1)

        # visuals['diff_map'] = diff_map

        # Handle saliency overlay visualization
        if self.saliency is None:
            saliency_map_vis = np.zeros_like(fake_B)  # Black placeholder
        else:
            saliency_np = self.saliency
            if saliency_np.ndim == 3:
                saliency_np = saliency_np[0]  # Take the first image from batch
            elif saliency_np.ndim == 4:
                saliency_np = saliency_np[0, 0]  # Handle [B, 1, H, W] case

            saliency_map_vis = overlay_saliency_on_pred(fake_B, saliency_np)


        visuals = OrderedDict([
            ('T1', real_T1),
            ('T2', real_T2),
            ('FLAIR',real_Flair),
            ('real_T1ce', real_B),
            ('Pred_T1ce', fake_B),
            ('Diff_Map', diff_map)
            # ('Saliency_Map', saliency_map_vis)
        ])

        return visuals


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        