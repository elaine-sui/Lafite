from torchvision import transforms as T

def build_transforms(resolution):
    transforms = T.Compose(
            [
                T.Resize(resolution, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop(resolution),
                T.ToTensor()
            ]
        )
    
    return transforms