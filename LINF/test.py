import os
import argparse
from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

import yaml
import lpips
import numpy as np
from PIL import Image
from tqdm import tqdm

import utils
import models
import datasets


def batched_predict(model, inp, coord, cell, temperature):
    with torch.no_grad():
        feat = model("gen_feat", inp=inp)

        #########################################
        #error checking if feat is not none.
        if feat is None:
          raise Exception("Feature generation failed; returned None.")
        #########################################
        _, h, w, _ = coord.shape
        row = 0
        preds = []
        while row < h:
            pred = model("query_rgb", inp=inp, feat=feat, coord=coord[:, row:row+256, :, :], cell=cell, temperature=temperature)
            preds.append(pred)
            row += 256
        pred = torch.cat(preds, dim=2)
    return pred


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None, window_size=0, scale_max=4,
              verbose=False, sample=0, detail=False, randomness=False, temperature=0, patch=False):
    model.eval()

    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    t = data_norm['inp']
    inp_sub = torch.FloatTensor(t['sub']).view(1, -1, 1, 1).cuda()
    inp_div = torch.FloatTensor(t['div']).view(1, -1, 1, 1).cuda()
    t = data_norm['gt']
    gt_sub = torch.FloatTensor(t['sub']).view(1, 1, -1).cuda()
    gt_div = torch.FloatTensor(t['div']).view(1, 1, -1).cuda()

    if eval_type is None:
        psnr_fn = utils.calc_psnr
    elif eval_type.startswith('div2k'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
    elif eval_type.startswith('benchmark'):
        scale = int(eval_type.split('-')[1])
        psnr_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
    else:
        raise NotImplementedError

    val_psnr = utils.Averager()
    if detail:
        # SSIM
        ssim_fn = utils.calculate_ssim
        val_ssim = utils.Averager()
        # LPIPS
        loss_fn_alex = lpips.LPIPS(net='alex').to('cuda')
        val_lpips = utils.Averager()
    if randomness:
        # Diversity
        val_diversity = utils.Averager()

    pbar = tqdm(loader, leave=False, desc='val')
    for idx, batch in enumerate(pbar):
        for k, v in batch.items():
            batch[k] = v.cuda()

        inp = (batch['inp'] - inp_sub) / inp_div
        # SwinIR Evaluation - reflection padding
        if window_size != 0:
            _, _, h_old, w_old = inp.size()
            h_pad = (h_old // window_size + 1) * window_size - h_old
            w_pad = (w_old // window_size + 1) * window_size - w_old
            inp = torch.cat([inp, torch.flip(inp, [2])], 2)[:, :, :h_old + h_pad, :]
            inp = torch.cat([inp, torch.flip(inp, [3])], 3)[:, :, :, :w_old + w_pad]
            
            coord = utils.make_coord((scale*(h_old+h_pad), scale*(w_old+w_pad)), flatten=False).unsqueeze(0).cuda()
            cell = batch['cell']
        else:
            h_pad = 0
            w_pad = 0
            
            coord = batch['coord']
            cell = batch['cell']
            
        if eval_bsize is None:
            with torch.no_grad():
                pred = model("rgb", inp=inp, coord=coord, cell=cell, temperature=temperature)
                if patch:   
                    # only evaluate central pixel of each patch when training
                    # pred is the fold tensor with shape [bs, 3, h*patch_size, w*patch_size]
                    # where h*w equals to the number of sampled HR coordinates
                    # TODO: find a clever method...
                    pred = pred.unfold(2, model.patch_size, model.patch_size).unfold(3, model.patch_size, model.patch_size)
                    bs, c, h, w, a, b = pred.shape
                    ps_square = model.patch_size*model.patch_size
                    pred = pred.contiguous().view(bs, c, h, w, ps_square).permute(0, 1, 4, 2, 3).contiguous().view(bs, c*ps_square, h, w)
                    pred = torch.concat([
                        pred[:, model.patch_size*model.patch_size//2].unsqueeze(1),
                        pred[:, model.patch_size*model.patch_size//2+model.patch_size*model.patch_size].unsqueeze(1),
                        pred[:, model.patch_size*model.patch_size//2+model.patch_size*model.patch_size*2].unsqueeze(1)]
                    , dim=1)
                    pred += F.grid_sample(inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)
        else:
            if randomness:
                preds = []
                for i in range(5):
                    pred = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), temperature)    # cell clip for extrapolation
                    pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
                    if patch:
                        # add residual back after getting all patches together
                        pred += F.interpolate(inp, pred.shape[-2:], mode='bilinear', align_corners=False)
                    preds.append(pred)
            else:
                pred = batched_predict(model, inp, coord, cell*max(scale/scale_max, 1), temperature)    # cell clip for extrapolation
                pred = pred[..., :batch['gt'].shape[-2], :batch['gt'].shape[-1]]
                if patch:
                    # add residual back after getting all patches together
                    pred += F.interpolate(inp, pred.shape[-2:], mode='bilinear', align_corners=False)

        if detail:
            if randomness:
                # SSIM: [0, 255], float, numpy
                preds_ssim = [torch.clamp(img * gt_div + gt_sub, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255. for img in preds]
                ssim_results = [ssim_fn(img, batch['gt'].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.) for img in preds_ssim]
                val_ssim.add(sum(ssim_results)/len(ssim_results), inp.shape[0])
                # LPIPS: [-1, 1], float
                preds_lpips = [torch.clamp(img, -1, 1) for img in preds]
                lpips_results = [loss_fn_alex(img, (batch['gt'] - gt_sub) / gt_div).mean().detach() for img in preds_lpips] # detach to avoid autograd on GPU
                val_lpips.add(sum(lpips_results)/len(lpips_results), inp.shape[0])
            else:
                # SSIM
                res = ssim_fn(torch.clamp(pred * gt_div + gt_sub, 0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy() * 255., batch['gt'].squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.)
                val_ssim.add(res.item(), inp.shape[0])
                # LPIPS
                d = loss_fn_alex(torch.clamp(pred, -1, 1), (batch['gt'] - gt_sub) / gt_div)
                val_lpips.add(d.mean().detach(), inp.shape[0])
        
        if randomness:
            # Diversity: [0, 255], uint8
            preds_diversity = [torch.round(torch.clamp(img * gt_div + gt_sub, 0, 1)*255.).unsqueeze(1) for img in preds]
            val_diversity.add(torch.std(torch.concat(preds_diversity, 1), dim=1).mean(), inp.shape[0])
            # PSNR
            preds_psnr = [torch.clamp(img * gt_div + gt_sub, 0, 1) for img in preds]
            psnr_results = [psnr_fn(img, batch['gt']) for img in preds_psnr]
            val_psnr.add(sum(psnr_results)/len(psnr_results), inp.shape[0])

            if idx < sample:
                img = (preds_psnr[0][0].permute(1, 2, 0) * 255.).cpu().numpy()
                img = Image.fromarray(img.round().astype(np.uint8), mode='RGB')
                img.save(os.path.join(save_path, '{}x{}_random.png'.format(800+idx+1, scale)))
        else:
            # PSNR
            pred = torch.clamp(pred * gt_div + gt_sub, 0, 1)
                
            if idx < sample:
                img = (pred[0].permute(1, 2, 0) * 255.).cpu().numpy()
                img = Image.fromarray(img.round().astype(np.uint8), mode='RGB')
                img.save(os.path.join(save_path, '{}x{}_zero.png'.format(800+idx+1, scale)))
            
            res = psnr_fn(pred, batch['gt'])
            val_psnr.add(res.item(), inp.shape[0])

        if verbose:
            pbar.set_description('psnr {:.4f}'.format(val_psnr.item()))
    
    if detail:
        result_dict = {'psnr': val_psnr.item(), 'ssim': val_ssim.item(), 'lpips': val_lpips.item()}
        if randomness:
            result_dict['diversity'] = val_diversity.item()
        return result_dict
    else:
        return val_psnr.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--window', default='0')
    parser.add_argument('--scale_max', default='30')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument("--detail", action='store_true')
    parser.add_argument("--randomness", action='store_true')
    parser.add_argument('--temperature', type=float, default=0)
    parser.add_argument('--patch', action='store_true')
    parser.add_argument('--name', type=str, default=None)

    args = parser.parse_args()

    if args.name is None:
        save_path = './sample'
    else:
        save_path = os.path.join('./sample', args.name)

    if args.sample > 0 and not os.path.isdir(save_path):
        print("create sample directory {}".format(save_path))
        if not os.path.isdir('./sample'):
            os.mkdir('./sample')
        os.mkdir(save_path)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    if args.patch:
        config['test_dataset']['wrapper']['name'] += '-patch'
        if 'downsampled' in config['test_dataset']['wrapper']['name']:
            config['test_dataset']['wrapper']['name'] += '-test'
        config['test_dataset']['wrapper']['args']['patch_size'] = model.patch_size

    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
        num_workers=8, pin_memory=True)

    res = eval_psnr(loader, model,
        data_norm=config.get('data_norm'),
        eval_type=config.get('eval_type'),
        eval_bsize=config.get('eval_bsize'),
        window_size=int(args.window),
        scale_max = int(args.scale_max),
        verbose=True,
        sample=args.sample,
        detail=args.detail,
        randomness=args.randomness,
        temperature=args.temperature,
        patch=args.patch)
    if args.detail:
        for key, val in res.items():
            print(key, ": {:.4f}".format(val))
    else:
        print('psnr: {:.4f}'.format(res))