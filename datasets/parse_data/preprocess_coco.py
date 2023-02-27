import os
from pathlib import Path
import pickle
import skimage.io as io
import numpy as np
from PIL import Image
from torchvision import transforms as T

from tqdm import tqdm

PREPROCESSED_ROOT = "/pasteur/u/esui/data/lafite/coco_preprocessed"


DATA_PATHS = ['/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_val.pkl',
                '/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_train.pkl',
              '/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_train+restval.pkl',
              '/pasteur/u/esui/data/coco/oscar_split_ViT-B_32_test.pkl'
             ]

def build_transforms(resolution):
    transforms = T.Compose(
            [
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor()
            ]
        )
    
    return transforms

def preproc_image(img_path, transforms):
    # Need to check dimensions!
    image = io.imread(img_path)
    image = transforms(Image.fromarray(image)).numpy()
    if image.max() <= 1.:
        image = (image * 255).astype('uint8')

    if image.shape[0] != 3:
        image = image.repeat(3, axis=0)
    
    # Need to permute
    image = np.transpose(image, (1, 2, 0))

    return image

def main():
    transforms = build_transforms(256)
    
    os.makedirs(PREPROCESSED_ROOT, exist_ok=True)
    
    for data_path in DATA_PATHS:
        print(f"Loading data from {data_path}")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
            
        images = all_data["images"]
        
        for img_id in tqdm(images):
            image_path = Path(images[img_id]["img_path"])
            # Separate path and only get the last filename and previous datasplit folder
            image_path_parts = image_path.parts[-2:]
            
            new_image_path = os.path.join(PREPROCESSED_ROOT, image_path_parts[0], 
                                          image_path_parts[1])
            os.makedirs(os.path.join(PREPROCESSED_ROOT, image_path_parts[0]), exist_ok=True)
            
            # Preprocess image
            image = preproc_image(image_path, transforms)
            
            # Save image
            io.imsave(new_image_path, image)
            
            all_data["images"][img_id]["img_path"] = new_image_path
            
        filename = Path(data_path).parts[-1]
        new_data_path = os.path.join(PREPROCESSED_ROOT, filename)
        with open(new_data_path, 'wb') as f:
            pickle.dump(all_data, f)

if __name__ == '__main__':
    main()
    