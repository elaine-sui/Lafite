import torch
import os
import pickle
import sys
import json
import random
import math
from pathlib import Path
from typing import Tuple, Optional, Union

import skimage.io as io
import clip
from PIL import Image

from omegaconf import OmegaConf
import numpy as np

from .enums import Modality
from .parse_data import TEXT_TO_IMG_GAP_PATH, TEXT_EMBED_MEAN, IMAGE_EMBED_MEAN
from .dataset_base import DatasetBase
from .builder import build_transforms

OLD_ROOT="/pasteur/u/esui/data/coco"

class ClipCocoDataset(DatasetBase):

    def __init__(self, cfg, split='train'):
        resolution = 256
        
        # Note: if ratio = 0, no text data is used (pseudo image-text pairs), if ratio = 1, 
        # gt image-text pairs used
            
        print("="*80)
        print("Data split: ", split)
        print("="*80)
        
        self.split = split
        
        self.cfg = cfg
        self.remove_modality_gap = self.cfg.data.remove_modality_gap
        self.remove_mean = self.cfg.data.remove_mean
        self.add_gaussian_noise = self.cfg.data.add_gaussian_noise
        
        data_path = self.get_data_path(cfg, split)
        self.normalize_prefix = cfg.data.normalize_prefix
        
        ###################
        print("=> Loading all_data pkl")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Number of images is %0d" % len(all_data["images"]))
        print("Number of captions is %0d" % len(all_data["captions"]))
        sys.stdout.flush()
        
        # {image_id: {"img_path": ..., "embed": ...}}
        self.images = all_data["images"]
        # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
        self.captions = all_data["captions"]
        
        ###################
        
        filepath = os.path.join(OLD_ROOT, f"{Path(data_path).parts[-1][:-4]}_tokens.pkl")
        with open(filepath, 'rb') as f:
            self.captions_tokens, self.caption_id_2_image_id, all_len = pickle.load(f)
        
        self.max_seq_len = min(int(all_len.mean() + all_len.std() * 10), int(all_len.max()))
    
        self.output_modality = self.cfg.decoder.modality
        
        # In testing, input modality must be opposite of output modality to evaluate cross-modal task
        if self.split == "test":
            if self.output_modality == Modality.Vision:
                self.input_modality = Modality.Language
            else:
                self.input_modality = Modality.Vision
        else:
            self.input_modality = self.cfg.encoder.modality
        
        # Get all caption and image ids
        self.img_ids = sorted(list(self.images.keys()))
        random.shuffle(self.img_ids)
        self.cap_ids = sorted(list(self.captions.keys()))
        random.shuffle(self.cap_ids)
        
        # Sample data
        if "train" in self.split and not OmegaConf.is_none(cfg.data, 'sample_frac'):
            img_sample_size = int(len(self.img_ids) * cfg.data.sample_frac)
            cap_sample_size = int(len(self.cap_ids) * cfg.data.sample_frac)
            self.img_ids = random.sample(self.img_ids, img_sample_size)
            self.cap_ids = random.sample(self.cap_ids, cap_sample_size)
        
        if self.input_modality == Modality.Language:
            self._raw_idx = self.cap_ids
        else:
            self._raw_idx = self.img_ids
        
        # Load modality gap
        with open(TEXT_TO_IMG_GAP_PATH, 'rb') as f:
            self.text_to_img_modality_gap = pickle.load(f)
            
        # Load means gap
        with open(TEXT_EMBED_MEAN, 'rb') as f:
            self.text_embed_mean = pickle.load(f)
            
        with open(IMAGE_EMBED_MEAN, 'rb') as f:
            self.image_embed_mean = pickle.load(f)
        
        ## Preprocess image to clip
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.preprocess_img = build_transforms(resolution)
        
        # _, self.preprocess_img = clip.load(self.cfg.encoder.clip_model_type, 
        #                                    device=device, jit=False)
        
        
        self.std = math.sqrt(0.016) # hyperparam from capdec paper
        
        super(ClipCocoDataset, self).__init__(
            name=cfg.data.dataset, 
            raw_shape=[len(self._raw_idx), 3, resolution, resolution],
            use_clip=True,
            ratio=0. if cfg.encoder.modality == Modality.Vision else 1.
        )
        
    def get_data_path(self, cfg, split):
        if split == 'train':
            data_path = cfg.data.train_data_path
        elif split == 'val':
            data_path = cfg.data.val_data_path
        elif split == 'test':
            data_path = cfg.data.test_data_path
        elif split == 'restval':
            data_path = cfg.data.restval_data_path
        elif split == 'train+restval':
            data_path = cfg.data.train_restval_data_path
        else:
            raise NotImplementedError(f"split {split} invalid")
            
        return data_path
        
    def _load_raw_labels(self):
        if self.input_modality == Modality.Language:
            return self.cap_ids
        else:
            return self.img_ids
    
    def _load_raw_image(self, item):
        if self.input_modality == Modality.Language:
            item = self.cap_ids[item]
            img_id = self.caption_id_2_image_id[item]
        else:
            img_id = self.img_ids[item]
        img_path = self.images[img_id]["img_path"]
        image = io.imread(img_path)
        
        if image.shape[0] != 256: # not preprocessed already
            image = self.preprocess_img(Image.fromarray(image)).numpy()
            if image.max() <= 1.:
                image = (image * 255).astype('uint8')

            if image.shape[0] != 3:
                image = image.repeat(3, axis=0)
        else:
            image = np.transpose(image, (2, 0, 1))
        
        return image

    def get_img_features(self, item):
        if self.input_modality == Modality.Language:
            item = self.cap_ids[item]
            img_id = self.caption_id_2_image_id[item]
        else:
            img_id = self.img_ids[item]
        img_prefix = self.images[img_id]["embed"].float()

        if self.normalize_prefix:
            img_prefix = img_prefix / img_prefix.norm(2, -1)
        
        # if self.remove_modality_gap: # simulates text embed
        #     # note: the gap was computed as img - text
        #     img_prefix -= self.text_to_img_modality_gap 
        #     img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)
        if self.remove_mean:
            img_prefix -= self.image_embed_mean
            img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)
        
        # if self.add_gaussian_noise:
        #     img_prefix += torch.randn(img_prefix.shape) * self.std
        #     img_prefix = torch.nn.functional.normalize(img_prefix, dim=-1)
            
        return img_prefix.squeeze().numpy()
    
    def get_pseudo_txt_features(self, item):
        # Only used when input modality is vision
        img_id = self.img_ids[item]
        pseudo_txt_prefix = self.images[img_id]["embed"].float()
        
        if self.normalize_prefix:
            pseudo_txt_prefix = pseudo_txt_prefix / pseudo_txt_prefix.norm(2, -1)
        
        if self.remove_modality_gap: # simulates text embed
            # note: the gap was computed as img - text
            pseudo_txt_prefix -= self.text_to_img_modality_gap 
            pseudo_txt_prefix = torch.nn.functional.normalize(pseudo_txt_prefix, dim=-1)
        elif self.remove_mean:
            pseudo_txt_prefix -= self.image_embed_mean
            pseudo_txt_prefix = torch.nn.functional.normalize(pseudo_txt_prefix, dim=-1)
        
        if self.add_gaussian_noise:
            pseudo_txt_prefix += torch.randn(pseudo_txt_prefix.shape) * self.std
            pseudo_txt_prefix = torch.nn.functional.normalize(pseudo_txt_prefix, dim=-1)
        
        return pseudo_txt_prefix.squeeze().numpy()
        

    def get_txt_features(self, item):
        if self.input_modality == Modality.Vision:
            text_prefix = self.get_pseudo_txt_features(item)
        else:
            item = self.cap_ids[item]
            text_prefix = self.captions[item]["embed"]

            if self.normalize_prefix:
                text_prefix = text_prefix.float()
                text_prefix = text_prefix / text_prefix.norm(2, -1)

            # if self.remove_modality_gap: # simulates image embed
            #     # note: the gap was computed as img - text
            #     text_prefix += self.text_to_img_modality_gap 
            #     text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)
            if self.remove_mean:
                text_prefix -= self.text_embed_mean
                text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)

            if self.add_gaussian_noise:
                text_prefix += torch.randn(text_prefix.shape) * self.std
                text_prefix = torch.nn.functional.normalize(text_prefix, dim=-1)

        if not isinstance(text_prefix, np.ndarray):
            text_prefix = text_prefix.squeeze().numpy()
        
        return text_prefix
    
## To get stuff:
# image_path = self.images[img_id]["img_path"]
# image_embed = self.images[img_id]["embed"]
# caption = self.captions[sent_id]["caption"]
# image_id_for_caption = self.captions[sent_id]["img_id"]
# caption_embed = self.captions[sent_id]["embed"]
