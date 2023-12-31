import torch
import numpy as np
import cv2

from point_segment.ViT import PlainVitModel
from point_segment.transforms.point_seg import Process
from point_segment.clicker import Clicker


def build_simpleclick_model(ckpt_p, device='cpu'):
    print(f'Load segmentation model from {ckpt_p}')
    kargs = {'use_disks': True, 'norm_radius': 5,
            'with_prev_mask': True, 'cpu_dist_maps': True}
    pointseg_model = PlainVitModel(**kargs)
    pointseg_model.load_state_dict(torch.load(ckpt_p, map_location='cpu'), strict=True)

    for param in pointseg_model.parameters():
        param.requires_grad = False

    pointseg_model.to(device)
    pointseg_model.eval()
    print(f'Warn up segmentation model...')
    input1_point_seg = torch.zeros((2, 4, 448, 448)).to(device)
    input2_point_seg = torch.zeros((2, 4, 3)).to(device)
    pointseg_model(input1_point_seg, input2_point_seg)
    return pointseg_model

def segment_img_with_builded_simpleclick(model, img: torch.Tensor, pre_mask: torch.Tensor, list_points: list[Clicker], base_size, device='cpu'):
    if pre_mask is None:
        pre_mask = torch.zeros(1, 1, *img.shape[2:])

    input_image = torch.cat((img, pre_mask), dim=1)
    seg_preprocess = Process(base_size, device)
    image_nd, clicks_lists, is_image_changed = seg_preprocess.preprocess(
        input_image, [list_points])

    with torch.no_grad():
        pred = model(image_nd, clicks_lists)['instances']
    pred = seg_preprocess.postprocess(pred)
    return pred 