"""
Modified by Yujin Oh
https://github.com/yjoh12/CXR-Segmentation-by-AdaIN-based-Domain-Adaptation-and-Knowledge-Distillation.git

Forked from StarGAN v2, Copyright (c) 2020-preeent NAVER Corp.
https://github.com/clovaai/stargan-v2.git
"""

import os
import argparse

from munch import Munch

import torch
from torch.backends import cudnn

from data_loader import get_train_loader, get_eval_loader, InputFetcher
from core.loss import compute_seg_loss, compute_d_loss, compute_g_loss
from core.model import build_model
from tqdm import tqdm
import copy

TensorLong = torch.cuda.LongTensor if torch.cuda.is_available() else torch.Tensor
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
DOMAIN_MASK = 2
DOMAIN_PAIRED = 1
DOMAIN_UNPAIRED = 0


def moving_average(model, model_test, beta=0.999):
    for param, param_test in zip(model.parameters(), model_test.parameters()):
        param_test.data = torch.lerp(param.data, param_test.data, beta)


def main(args):

    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # data loader
    train_loader = get_train_loader(root=args.train_img_dir, which='source_mask', img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    ref_loader = get_train_loader(root=args.train_img_dir, which='reference', img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers, args=args)
    val_loader = get_eval_loader(root=args.val_img_dir, img_size=args.img_size, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers,args=args)

    # model
    GPU_NUM = args.gpu
    device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    nets, nets_ema = build_model(args)
    for name, module in nets.items():
        module.to(device)
    for name, module in nets_ema.items():
        module.to(device)

    # optimizer
    optims = Munch()
    lr_scheduler = Munch()
    for net in nets.keys():
        optims[net] = torch.optim.Adam(params=nets[net].parameters(), lr=args.f_lr if net == 'mapping_network' else args.lr, betas=[args.beta1, args.beta2], weight_decay=args.weight_decay)
        lr_scheduler[net] = torch.optim.lr_scheduler.MultiStepLR(optims[net], milestones=[int(args.total_iters*0.55)-args.resume_iter, int(args.total_iters*0.7)-args.resume_iter], gamma=0.1) 
    
    # prepare training
    initial_lambda_ds = args.lambda_ds 
    path_src = os.path.join(args.val_img_dir, args.domain1)
    loader_src_mask = get_eval_loader(root=path_src, img_size=args.img_size, batch_size=args.batch_size, drop_last=False, args=args)  
    x_src_ref = loader_src_mask.dataset.__getitem__(0).to(device).unsqueeze(0) #1
    y_trg_src = torch.tensor([DOMAIN_PAIRED]).to(device)
    y_trg_msk = torch.tensor([DOMAIN_MASK]).to(device)
    
    # self-supervised learning
    if args.flag_self:
        _load_checkpoint_pretrain(nets, step=args.resume_iter, dir=args.checkpoint_dir,)   
        nets_freeze = copy.deepcopy(nets)
        s_src = nets_freeze.style_encoder(x_src_ref, y_trg_src)
    
    # train
    print('Start training...')
    log_loss = []
    trainer = tqdm(range(args.resume_iter, args.total_iters))
    fetcher = InputFetcher(train_loader, ref_loader, args.latent_dim, 'train_mask')

    for i in trainer:
        
        # save model checkpoints
        if (i+1) % args.eval_every == 1:
            _save_checkpoint(nets_ema, step=i, dir=args.checkpoint_dir, flag_self=args.flag_self)
                
        # print out log info
        if (i+1) % args.print_every == 1:
            log = "Iter [%i/%i], " % (i, args.total_iters)
            all_losses = dict()
            if args.flag_self:
                try:
                    log += loss_self.item()
                except:
                    log += 'loss_self is not calcuated yet'
            else:
                try:
                    for loss, prefix in zip([d_losses_ref, g_losses_ref, g_loss_seg_ref], ['D/ref_', 'G/ref_', 'Seg/ref_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                        all_losses['G/lambda_ds'] = args.lambda_ds
                    for loss, prefix in zip([d_losses_latent, g_losses_latent, g_loss_seg_latent], ['D/latent_', 'G/latent_', 'Seg/latent_']):
                        for key, value in loss.items():
                            all_losses[prefix + key] = value
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                except:
                    log += 'loss_latent, loss_ref is not calcuated yet'
            print(log)
        
        # fetch x and y
        inputs = next(fetcher)
        x_real, y_org = inputs.x_src, inputs.y_src
        x_ref, x_ref2, y_trg = inputs.x_ref, inputs.x_ref2, inputs.y_ref
        
        ## PASS TRAINING WHEN:
        # MASK2EVERYDOMAIN
        if (y_org[0].item() == DOMAIN_MASK): 
            continue 
        # UNPAIREDl2MASK
        if (not args.flag_self) & (y_org[0].item() == DOMAIN_UNPAIRED) & (y_trg[0].item() == DOMAIN_MASK): 
            continue

        # decay weight for diversity sensitive loss
        if args.lambda_ds>0:
            args.lambda_ds -= (initial_lambda_ds / args.ds_iter)
                        
        # fetch msk
        z_trg, z_trg2, z_self = inputs.z_trg, inputs.z_trg2, inputs.z_self
        x_msk, y_msk = inputs.x_msk, inputs.y_msk
        
        # prepare mask
        x_msk = torch.div(x_msk+1, 2, rounding_mode='trunc').type(Tensor)
        if (y_trg[0].item() == DOMAIN_MASK):
            x_ref, x_ref2 = x_msk, x_msk
            x_ref = torch.eye(args.seg_class)[:,x_ref.type(torch.long)].transpose(0,1).type(Tensor)
            x_ref2 = torch.eye(args.seg_class)[:,x_ref2.type(torch.long)].transpose(0,1).type(Tensor)
        
        ### segmentation
        if (y_org[0].item() == (DOMAIN_PAIRED)) & (y_trg[0].item() == DOMAIN_MASK):
            
            # latent-guided generation
            g_loss, g_loss_seg_ref = compute_seg_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg, msk=x_msk)
            _reset_grad(optims)
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            # reference-guided generation
            g_loss, g_loss_seg_latent = compute_seg_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref, msk=x_msk)
            _reset_grad(optims)
            g_loss.backward()
            optims.generator.step()    


        ### domain addaptation
        if (y_trg[0].item() != DOMAIN_MASK):        

            # latent-guided discriminator
            d_loss, d_losses_latent = compute_d_loss(nets, args, x_real, y_org, y_trg, z_trg=z_trg)
            _reset_grad(optims)
            d_loss.backward()
            optims.discriminator.step()

            # reference-guided discriminator
            d_loss, d_losses_ref = compute_d_loss(nets, args, x_real, y_org, y_trg, x_ref=x_ref)
            _reset_grad(optims)
            d_loss.backward()
            optims.discriminator.step()

            # latent-guided generation
            g_loss, g_losses_latent = compute_g_loss(nets, args, x_real, y_org, y_trg, z_trgs=[z_trg, z_trg2], x_refs=[x_ref, x_ref2], msk=x_msk, z_self=z_self)
            _reset_grad(optims)
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            optims.style_encoder.step()

            # reference-guided generation
            g_loss, g_losses_ref = compute_g_loss(nets, args, x_real, y_org, y_trg, x_refs=[x_ref, x_ref2], msk=x_msk)
            _reset_grad(optims)
            g_loss.backward()
            optims.generator.step()               
            
            
        ## self-supervised learning
        if args.flag_self & (y_trg[0].item() == DOMAIN_MASK):

            with torch.no_grad():

                if (y_org[0].item() == 0): # A2M
                    s_src = nets.style_encoder(x_src_ref, y_trg_src)
                    x_A2N = nets.generator(x_real, y_org, y_trg_src, s_src) 
                    x_A2N2M = nets.generator(x_A2N, y_trg_src, y_trg_msk, [], seg=True)
                    weight = args.lambda_self
                    
                if (y_org[0].item() == 1): # N2M
                    x_A2N2M = nets.generator(x_real, y_org, y_trg_msk, [], seg=True)
                    weight = args.lambda_self_inter
        
            # A2M
            s_trg_src = nets.mapping_network(z_self, y_trg+1) 
            x_A2M = nets.generator(x_real, y_org, y_trg_msk, [], seg=True, self_cons=s_trg_src)
            
            # L1
            loss_self = torch.mean(torch.abs(x_A2N2M - x_A2M))
            g_loss = weight * loss_self

            _reset_grad(optims)
            g_loss.backward()
            optims.generator.step()
            optims.mapping_network.step()
            
            
        # learning rate
        for net in nets.keys():
            lr_scheduler[net].step()
        
        # compute moving average of network parameters
        moving_average(nets.generator, nets_ema.generator, beta=0.999)
        moving_average(nets.mapping_network, nets_ema.mapping_network, beta=0.999)
        moving_average(nets.style_encoder, nets_ema.style_encoder, beta=0.999)
    

def _save_checkpoint(nets, step, dir='', flag_self=False):
    outdict = {}
    for ckptio in nets:
        for name, module in nets.items():
            if flag_self:
                if not ((name == 'generator') | (name == 'mapping_network')):
                    continue
            outdict[name] = module.state_dict()
    torch.save(outdict, '%s/%06d_nets.ckpt'%(dir, step))    
    
    
def _load_checkpoint_pretrain(nets, step, dir=''):
    module_dict = torch.load('%s/%06d_nets.ckpt'%(dir, step))
    for name, module in nets.module_dict.items():
        module.load_state_dict(module_dict[name])
        
            
def _reset_grad(optims):
    for optim in optims.values():
        optim.zero_grad()        
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--gpu', type=int, default=0)

    # experiment condition
    parser.add_argument('--flag_3class', type=bool, default=True)
    parser.add_argument('--resize', type=bool, default=False)
    parser.add_argument('--save_tag', type=str, default='_large')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for D, E and G') #w_Back 1&lr 1e-3

    # self consistency
    parser.add_argument('--flag_self', type=bool, default=False)
    parser.add_argument('--lambda_self', type=float, default=10) 
    parser.add_argument('--lambda_self_inter', type=float, default=1) 
     
    # segmentation
    parser.add_argument('--lambda_seg', type=float, default=5)
    parser.add_argument('--post_process', type=bool, default=True)
    
    # evaluation settings
    parser.add_argument('--mode', type=str, default='eval', choices=['train', 'test', 'eval'], help='This argument is used in solver')
    parser.add_argument('--total_iters', type=int, default=40000, help='Number of total iterations')
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--eval_every', type=int, default=4000)
    parser.add_argument('--resume_iter', type=int, default=0, help='Iterations to resume training/testing')

    # model arguments
    parser.add_argument('--img_size', type=int, default=256,
                        help='Image resolution')
    parser.add_argument('--latent_dim', type=int, default=4,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, 
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=16, 
                        help='Style code dimension')

    # weight for objective functions
    parser.add_argument('--lambda_cyc', type=float, default=2, 
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 *regularization')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1, 
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--ds_iter', type=int, default=40000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=6,
                        help='Batch size for validation')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=6,
                        help='Number of generated images per domain during sampling')

    # misc
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=407,
                        help='Seed for random number generator')
    
    # customize
    args = parser.parse_args()
    root_dir = "/media/yujins/E/"
    save_dir = "v1_"
    parser.add_argument('--num_domains', type=int, default=3) # Abnormal, Normal, Mask
    parser.add_argument('--domain0', type=str, default='a)Abnormal') # Unpaired domain
    parser.add_argument('--domain1', type=str, default='b)Normal') # Paired image domain
    parser.add_argument('--domain2', type=str, default='c)Mask') # Paired mask domain
    parser.add_argument('--input_dim', type=int, default=1) # Input image dimension
    parser.add_argument('--seg_class', type=int, default=2) # background, lung
    parser.add_argument('--val_img_dir', type=str, default=root_dir+'Pneumonia_Collection/Public_Korea_Val_large')
    parser.add_argument('--train_img_dir', type=str, default=root_dir+'Pneumonia_Collection/Public_large') 
    if args.flag_self:
        args.resume_iter = 1000000
        args.total_iters = args.resume_iter + args.total_iters
        args.eval_every = int(args.eval_every/5) 
        
    # directories
    parser.add_argument('--checkpoint_dir', type=str, default=save_dir+'expr/checkpoints', help='Directory for saving network checkpoints')
    parser.add_argument('--result_dir', type=str, default=save_dir+'expr/results', help='Directory for saving generated images and videos')
    
    # start training
    args = parser.parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.result_dir, exist_ok=True)
    main(args)
