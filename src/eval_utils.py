import random
import os
from pathlib import Path

import numpy as np
import torch

from transformers import AutoProcessor, CLIPModel

def tensor2numpy(tensor:torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().numpy()
    tensor = (tensor * 255).astype(np.uint8)
    return tensor

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
    local_imgs = []
    for key in preds.keys():
        if 'global_image' in key and 'norm' not in key:
            global_img = tensor2numpy(preds[key][0])
        if 'local' in key and 'image' in key and 'norm' not in key:
            local_img = tensor2numpy(preds[key][0])
            local_imgs.append(local_img)
    return [global_img], local_imgs

def CLIP_score(model, processor, text, global_image, local_images):
    model.eval()
    global_input = processor(text=[text], images=global_image, return_tensors='pt', padding=True)
    global_outputs = model(**global_input)
    local_input = processor(text=[text], images=local_images, return_tensors='pt', padding=True)
    local_outputs = model(**local_input)
    return global_outputs.logits_per_text[0][0], local_outputs.logits_per_text.mean()