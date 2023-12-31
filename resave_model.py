

import torch
from point_segment.ViT import PlainVitModel

kargs = {'use_disks': True, 'norm_radius': 5,
         'with_prev_mask': True, 'cpu_dist_maps': True}

pointseg_model = PlainVitModel(**kargs)
state_dict = torch.load('./pretrained_models/cocolvis_vit_huge.pth', map_location='cpu')
pointseg_model.load_state_dict(state_dict['state_dict'], strict=True)
torch.save(pointseg_model.state_dict(), './pretrained_models/vit_huge.pth')