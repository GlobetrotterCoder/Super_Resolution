import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np

import models
from models import register
from utils import make_coord

@register('linf')
class LINF(nn.Module):

    class WaveletTransformLayer(nn.Module):
        def __init__(self, wavelet_name='db3'):
            super().__init__()
            self.wavelet = pywt.Wavelet(wavelet_name)

        def forward(self, x):
            # Detach tensor from the computation graph, move to CPU, and convert to NumPy array
            x_np = x.detach().cpu().numpy()
            # Perform the wavelet transform
            coeffs = pywt.dwt2(x_np, wavelet=self.wavelet, mode='zero')
            cA, (cH, cV, cD) = coeffs
            # Convert back to tensors and move to the original device
            cA = torch.from_numpy(cA).to(x.device)
            cH = torch.from_numpy(cH).to(x.device)
            cV = torch.from_numpy(cV).to(x.device)
            cD = torch.from_numpy(cD).to(x.device)
            return torch.cat((cA, cH, cV, cD), dim=1)

    def __init__(self, encoder_spec, imnet_spec=None, flow_layers=10, num_layer=3, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.wavelet_transform = self.WaveletTransformLayer()

        layers = []
        layers.append(nn.Conv2d(hidden_dim*4, hidden_dim, 1))  # Adjust input channels according to wavelet output
        layers.append(nn.ReLU())

        for i in range(num_layer-1):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU())

        layers.append(nn.Conv2d(hidden_dim, flow_layers*3*2, 1))
        self.layers = nn.Sequential(*layers)

        self.imnet = models.make(imnet_spec, args={'flow_layers': flow_layers})
        
    def gen_feat(self, inp):
        feat = self.encoder(inp)
        feat = self.wavelet_transform(feat)
        return feat

    def query_log_p(self, inp, feat, coord, cell, gt):
        gt_residual = gt - F.grid_sample(inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        areas = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx
                coord_[:, :, :, 1] += vy * ry
                coord_.clamp_(-1, 1)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)

        # Upsample feat to match area and tot_area resolution
        feat_upsampled = F.interpolate(feat, size=(48, 48), mode='bilinear', align_corners=False)
        
        # Expand area dimensions to match feat_upsampled channel dimension
        expanded_areas = [a.unsqueeze(1).expand(-1, feat_upsampled.size(1), -1, -1) for a in areas]
        features = torch.cat([a / tot_area.unsqueeze(1) * feat_upsampled for a in expanded_areas], dim=1)
        affine_info = self.layers(features)

        bs, w, h, _ = coord.shape
        z, log_p = self.imnet(gt_residual.permute(0, 2, 3, 1).contiguous().view(bs * w * h, -1), affine_info.view(bs * w * h, -1))

        return log_p


    def query_rgb(self, inp, feat, coord, cell, temperature=0):
        # Initial residual computation
        residual = inp - F.grid_sample(inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)

        # Field radius calculations (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        # Generating feature coordinates
        feat_coord = make_coord(feat.shape[-2:], flatten=False).cuda() \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(feat.shape[0], 2, *feat.shape[-2:])

        areas = []
        wavelet_features = []
        for vx in [-1, 1]:
            for vy in [-1, 1]:
                coord_ = coord.clone()
                coord_[:, :, :, 0] += vx * rx
                coord_[:, :, :, 1] += vy * ry
                coord_.clamp_(-1, 1)
                q_coord = F.grid_sample(feat_coord, coord_.flip(-1), mode='nearest', align_corners=False)
                rel_coord = coord.permute(0, 3, 1, 2) - q_coord

                area = torch.abs(rel_coord[:, 0, :, :] * rel_coord[:, 1, :, :])
                areas.append(area + 1e-9)

        # Summing the total area for normalization
        tot_area = torch.stack(areas).sum(dim=0)

        # Upsample feat to match area and tot_area resolution
        feat_upsampled = F.interpolate(feat, size=(48, 48), mode='bilinear', align_corners=False)

        # Concatenating and normalizing wavelet features
        expanded_areas = [a.unsqueeze(1).expand(-1, feat_upsampled.size(1), -1, -1) for a in areas]
        weighted_feats = [feat_upsampled * a for a in expanded_areas]

        # Aggregate features
        aggregated_features = torch.cat(weighted_feats, dim=1) / tot_area.unsqueeze(1)

        # Generate predictions
        affine_info = self.layers(aggregated_features)
        bs, w, h, _ = coord.shape
        pred = self.imnet.inverse((torch.randn((bs * w * h, 3)).cuda()) * temperature, affine_info.view(bs * w * h, -1))
        pred = pred.view(bs, w, h, -1).permute(0, 3, 1, 2).contiguous()

        # Adding the initial residual back to prediction
        pred += F.grid_sample(inp, coord.flip(-1), mode='bilinear', padding_mode='border', align_corners=False)

        return pred


    def forward(self, op, inp=None, feat=None, coord=None, cell=None, gt=None, temperature=0):

        if feat is None:
           feat = self.gen_feat(inp)
           
        # op: "query_log_p", "query_rgb", "log_p", "rgb", "gen_feat"
        if op == "query_log_p":
            return self.query_log_p(inp, feat, coord, cell, gt)
        if op == "query_rgb":
            return self.query_rgb(inp, feat, coord, cell, temperature)
        if op == "log_p":
            return self.query_log_p(inp, feat, coord, cell, gt)  # corrected call to query_log_p
        if op == "rgb":
            return self.query_rgb(inp, feat, coord, cell, temperature)  # corrected call to query_rgb
        if op == "gen_feat":
            return self.gen_feat(inp)
