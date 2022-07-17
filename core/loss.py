"""
Modified by Yujin Oh
https://github.com/yjoh12/CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation.git

Forked from StarGAN v2, Copyright (c) 2020-preeent NAVER Corp.
https://github.com/clovaai/stargan-v2.git
"""

import torch
import torch.nn as nn
from munch import Munch
import torch.nn.functional as F

TensorLong = torch.cuda.LongTensor if torch.cuda.is_available() else torch.Tensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
DOMAIN_MASK = 2
DOMAIN_PAIRED = 1
DOMAIN_UNPAIRED = 0


seg_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.7, 1.0]).type(Tensor)).cuda()


def compute_seg_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, msk=None):
    
    x_fake = nets.generator(x_real, y_org, y_trg, s=[], masks=masks, seg=True)
    loss_seg = seg_loss(x_fake, msk.squeeze(1).type(TensorLong))

    # seg loss
    loss = args.lambda_seg * loss_seg 
    
    return loss, Munch(seg=loss_seg.item())


def compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=None, x_ref=None, masks=None, msk=None):
    assert (z_trg is None) != (x_ref is None)
    # with real images
    x_real.requires_grad_()
    out = nets.discriminator(x_real, y_org)
    loss_real = adv_loss(out, 1)
    loss_reg = r1_reg(out, x_real)

    # with fake images
    with torch.no_grad():
        if z_trg is not None:
            s_trg = nets.mapping_network(z_trg, y_trg)
        else:  # x_ref is not None
            s_trg = nets.style_encoder(x_ref, y_trg)
        if (y_trg[0].item() == DOMAIN_MASK):
            x_fake = nets.generator(x_real, y_org, y_trg, s=[], masks=masks, seg=True)
        else:
            x_fake = nets.generator(x_real, y_org, y_trg, s_trg, masks=masks)

    out = nets.discriminator(x_fake, y_trg)
    loss_fake = adv_loss(out, 0)

    loss = loss_real + loss_fake + args.lambda_reg * loss_reg
    return loss, Munch(real=loss_real.item(),
                    fake=loss_fake.item(),
                    reg=loss_reg.item())


def compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=None, x_refs=None, masks=None, msk=None, z_self=None):
    # assert (z_trgs is None) != (x_refs is None)
    if z_trgs is not None:
        z_trg, z_trg2 = z_trgs
    if x_refs is not None:
        x_ref, x_ref2 = x_refs

    # adversarial loss
    if z_trgs is not None:
        s_trg = nets.mapping_network(z_trg, y_trg)
    else:
        s_trg = nets.style_encoder(x_ref, y_trg)

    # adv loss
    if (y_trg[0].item() == DOMAIN_MASK): # y_trg : 2 'Mask' 
        x_fake = nets.generator(x_real, y_org, y_trg, s=[], masks=masks, seg=True)
    else:
        x_fake = nets.generator(x_real, y_org, y_trg, s_trg, masks=masks)
    out = nets.discriminator(x_fake, y_trg)
    loss_adv = adv_loss(out, 1)

    # style reconstruction loss
    s_pred = nets.style_encoder(x_fake, y_trg)
    loss_sty = torch.mean(torch.abs(s_pred - s_trg))

    # cycle-consistency loss
    if (not (y_trg[0].item() == DOMAIN_MASK)): # not y_trg : 2 'Mask' 
        masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
        s_org = nets.style_encoder(x_real, y_org)  
        x_rec = nets.generator(x_fake, y_trg, y_org, s_org, masks=masks)
        loss_cyc = torch.mean(torch.abs(x_rec - x_real))
    else:
        loss_cyc = torch.tensor(0)
        
    # diversity sensitive loss
    if (not (y_trg[0].item() == DOMAIN_MASK)): # not y_trg : 2 'Mask' 
        if z_trgs is not None:
            s_trg2 = nets.mapping_network(z_trg2, y_trg)
        else:
            s_trg2 = nets.style_encoder(x_ref2, y_trg)
        x_fake2 = nets.generator(x_real, y_org, y_trg, s_trg2, masks=masks)
        x_fake2 = x_fake2.detach()
        loss_ds = torch.mean(torch.abs(x_fake - x_fake2))
    else:
        loss_ds = torch.tensor(0)

    # total loss
    loss = loss_adv + args.lambda_sty * loss_sty + args.lambda_cyc * loss_cyc - args.lambda_ds * loss_ds

    list_loss = Munch(adv=loss_adv.item(),
                sty=loss_sty.item(),
                ds=loss_ds.item(),
                cyc=loss_cyc.item(),
                )

    return loss, list_loss

def adv_loss(logits, target):
    assert target in [1, 0]
    targets = torch.full_like(logits, fill_value=target)
    loss = F.binary_cross_entropy_with_logits(logits, targets)
    return loss


def r1_reg(d_out, x_in):
    # zero-centered gradient penalty for real images
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg
