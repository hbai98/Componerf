import random
import os
from pathlib import Path

import numpy as np
import torch

from transformers import AutoProcessor, CLIPModel

def get_view_direction(thetas, phis, overhead, front):
    #                   phis [B,];          thetas: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(thetas.shape[0], dtype=torch.long)
    # first determine by phis

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res


def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

def get_label_text(data, main_text):
    label_texts = []
    # ref_text = self.nerf.cfg.guide.text
    ref_text = main_text
    for d in ['front', 'side', 'back', 'side', 'overhead', 'bottom']:
        text = f"{ref_text}, {d} view"
        label_texts.append(text)
    label_text = label_texts[data['dir'][0]]
    return label_text

def get_global_local_imgs(preds):
    print(preds.shape)
    local_imgs = []
    for key in preds.keys():
        if 'global_image' in key and 'norm' not in key:
            global_img = tensor2numpy(preds[key][0])
        if 'local' in key and 'image' in key and 'norm' not in key:
            local_img = tensor2numpy(preds[key][0])
            local_imgs.append(local_img)
    return [global_img], local_imgs

def CLIP_score(model, processor, text, image):
    model.eval()
    input = processor(text=[text], images=image, return_tensors='pt', padding=True)
    outputs = model(**input)
    # local_input = processor(text=[text], images=local_images, return_tensors='pt', padding=True)
    # local_outputs = model(**local_input)
    return outputs.logits_per_text[0][0]

