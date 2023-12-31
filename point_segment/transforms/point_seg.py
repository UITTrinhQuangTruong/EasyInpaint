import torch
import torch.nn.functional as F
from point_segment.clicker import Clicker
from typing import Optional

class BaseTransform:
    def __init__(self) -> None:
        self.image_changed = False
    
    def __call__(self, image_nd, clicks_lists):
        return image_nd, clicks_lists
    
    def inv(self, prob_map):
        return prob_map

    def reset(self):
        pass
        

class Resize(BaseTransform):
    def __init__(self, base_size=[448, 448],
                 skip_clicks=-1,
                 ):
        super().__init__()
        self.base_size = base_size 
        self.skip_clicks = skip_clicks

        self._input_image_shape = None
        self._prev_probs = None
        self._roi_image = None

        self.rmin, self.rmax, self.cmin, self.cmax = [None]*4


    def __call__(self, image_nd, clicks_lists):
        assert image_nd.shape[0] == 1 and len(clicks_lists) == 1
        self.image_changed = False

        if self.rmin is None:
            self.rmin, self.rmax, self.cmin, self.cmax = 0, image_nd.shape[2] - 1, 0, image_nd.shape[3] - 1
            self.image_changed = True
        crop_height, crop_width = self.base_size 


        clicks_list = clicks_lists[0]
        if len(clicks_list) <= self.skip_clicks:
            return image_nd, clicks_lists

        # Rescale clicks 
        transformed_clicks = []
        for click in clicks_list:
            new_c = crop_width * (click.x - self.cmin) / (self.cmax - self.cmin + 1)
            new_r = crop_height * (click.y - self.rmin) / (self.rmax - self.rmin + 1)
            transformed_clicks.append(Clicker(new_c, new_r, click.positive, click.indx))

        # resize image
        with torch.no_grad():
            resized_image = F.interpolate(image_nd, size=(crop_height, crop_width),
                                                        mode='bilinear', align_corners=True)
        return resized_image, [transformed_clicks]
    
    def inv(self, prob_map):
        if self.rmin is None:
            self._prev_probs = prob_map.cpu().numpy()
            return prob_map

        prob_map = torch.nn.functional.interpolate(prob_map, size=(self.rmax - self.rmin + 1, self.cmax - self.cmin + 1),
                                                   mode='bilinear', align_corners=True)

        if self._prev_probs is not None:
            new_prob_map = torch.zeros(
                *self._prev_probs.shape, device=prob_map.device, dtype=prob_map.dtype)
            new_prob_map[:, :, self.rmin:self.rmax + 1, self.cmin:self.cmax + 1] = prob_map
        else:
            new_prob_map = prob_map

        self._prev_probs = new_prob_map.cpu().numpy()

        return new_prob_map
    
    def reset(self):
        self._input_image_shape = None
        self._prev_probs = None
        self.image_changed = False

        self.rmin, self.rmax, self.cmin, self.cmax = [None]*4
        
class SigmoidForPred(BaseTransform):
    def inv(self, prob_map):
        return torch.sigmoid(prob_map)

class AddHorizontalFlip(BaseTransform):
    def __call__(self, image_nd, clicks_lists):
        assert len(image_nd.shape) == 4
        image_nd = torch.cat([image_nd, torch.flip(image_nd, dims=[3])], dim=0)

        image_width = image_nd.shape[3]
        clicks_lists_flipped = []
        for clicks_list in clicks_lists:
            clicks_list_flipped = [Clicker(image_width - click.x - 1, click.y, click.positive, click.indx)
                                   for click in clicks_list]
            clicks_lists_flipped.append(clicks_list_flipped)
        clicks_lists = clicks_lists + clicks_lists_flipped

        return image_nd, clicks_lists
    
    def inv(self, prob_map):
        assert len(prob_map.shape) == 4 and prob_map.shape[0] % 2 == 0
        num_maps = prob_map.shape[0] // 2
        prob_map, prob_map_flipped = prob_map[:num_maps], prob_map[num_maps:]

        return 0.5 * (prob_map + torch.flip(prob_map_flipped, dims=[3]))



class ToTensor(BaseTransform):
    def __init__(self, device='cpu') -> None:
        super().__init__()
        self.device = device
        self.net_clicks_limit = None

    def __call__(self, image_nd, clicks_lists):
        if not torch.is_tensor(image_nd):
            image_nd = torch.tensor(image_nd, device=self.device)
        
        total_clicks = []
        num_pos_clicks = [sum(x.is_positive for x in clicks_list)
                          for clicks_list in clicks_lists]
        num_neg_clicks = [len(clicks_list) - num_pos for clicks_list,
                          num_pos in zip(clicks_lists, num_pos_clicks)]
        num_max_points = max(num_pos_clicks + num_neg_clicks)
        if self.net_clicks_limit is not None:
            num_max_points = min(self.net_clicks_limit, num_max_points)
        num_max_points = max(1, num_max_points)

        for clicks_list in clicks_lists:
            clicks_list = clicks_list[:self.net_clicks_limit]
            pos_clicks = [
                click.value for click in clicks_list if click.is_positive]
            pos_clicks = pos_clicks + \
                (num_max_points - len(pos_clicks)) * [(-1, -1, -1)]

            neg_clicks = [
                click.value for click in clicks_list if not click.is_positive]
            neg_clicks = neg_clicks + \
                (num_max_points - len(neg_clicks)) * [(-1, -1, -1)]
            total_clicks.append(pos_clicks + neg_clicks)
        points_nd = torch.tensor(total_clicks, device=self.device)
        return image_nd, points_nd 


class Process:
    def __init__(self, base_size, device) -> None:
        self.base_size = base_size
        self.transforms = [SigmoidForPred(), AddHorizontalFlip(), ToTensor(device=device)]
    
    def preprocess(self, image_nd, points_nd):
        is_image_changed = False
        for t in self.transforms:
            image_nd, points_nd = t(image_nd, points_nd)
            is_image_changed |= t.image_changed
        return image_nd, points_nd, is_image_changed

    def postprocess(self, prob_map):
        for t in self.transforms[::-1]:
            prob_map = t.inv(prob_map) 
        return prob_map

    def reset_process(self):
        for t in self.transforms:
            try:
                t.reset()
            except Exception as e:
                print(e)
                return False
        
        return True