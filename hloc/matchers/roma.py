import torch
import torch.nn.functional as F
from romatch import roma_outdoor

from ..utils.base_model import BaseModel


class RoMa(BaseModel):
    default_conf = {}
    required_inputs = [
        "image0",
        "keypoints0",
        "image1",
        "keypoints1",
    ]

    def _init(self, conf):
        self.net = roma_outdoor(device=None)
        self.discard_match_th = conf['discard_match_th']
        self.device = conf['device']

    def _forward(self, data):
        im0_size, im1_size = data["image_size0"][0].cpu().numpy(), data["image_size1"][0].cpu().numpy()
        im0_coarse, im1_coarse = data["image0"][0].to(self.device), data["image1"][0].to(self.device)
        im0_fine, im1_fine = data["image0"][1].to(self.device), data["image1"][1].to(self.device)
        h0, w0 = im0_size[0], im0_size[1]
        h1, w1 = im1_size[0], im1_size[1]

        warp, certainty = self.net.match(im0_coarse, im1_coarse,
                                         im_A_high_res=im0_fine,
                                         im_B_high_res=im1_fine)

        if self.net.symmetric:
            warp_width = warp.shape[-2] // 2
            warp = warp[0,:,:warp_width,:]
            certainty = certainty[0,:,:warp_width]
        else:
            warp = warp[0]
            certainty = certainty[0]
        
        kpts0, kpts1 = data["keypoints0"][0].to(self.device), data["keypoints1"][0].to(self.device)
        kpts0 = torch.stack((2 * (kpts0[...,0] - 0.5 * w0) / w0, 2 * (kpts0[...,1] - 0.5 * h0) / h0), dim=-1)
        kpts1 = torch.stack((2 * (kpts1[...,0] - 0.5 * w1) / w1, 2 * (kpts1[...,1] - 0.5 * h1) / h1), dim=-1)
        matches0, matches1 = self.net.match_keypoints(kpts0, kpts1, warp, certainty, return_inds=True)

        certainty = F.grid_sample(certainty[None, None, ...], kpts0[None, None], 
                                  align_corners=False, mode="bilinear")[0, 0, 0]

        mean_certainty = certainty[matches0].mean().item()

        out_matches0 = -1 * torch.ones_like(kpts0[:,0]).to(torch.long)
        out_matches1 = -1 * torch.ones_like(kpts1[:,0]).to(torch.long)
        
        out_matches0[matches0] = matches1
        out_matches1[matches1] = matches0

        if mean_certainty < self.discard_match_th:
            empty = torch.zeros_like(out_matches0).bool()
            empty1 = torch.zeros_like(out_matches1).bool()
            out_matches0 = out_matches0[empty]
            out_matches1 = out_matches1[empty1]
            certainty = certainty[empty]
        
        out= {}
        out["matches0"] = out_matches0[None]
        out["matches1"] = out_matches0[None]
        out["matching_scores0"] = certainty[None]
        
        return out
