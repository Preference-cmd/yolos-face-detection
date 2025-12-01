from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoImageProcessor
from albumentations.pytorch import ToTensorV2   

import cv2
import torch
import albumentations as A


class FaceData(Dataset):
    def __init__(self, processor, image_dir, label_dir, transform = None, size = 640) -> None:
        """
        Args:
            processor: The image processor to use.
            image_dir: The directory containing the images.
            label_dir: The directory containing the labels.
            transform: The transform to apply to the images.
        """

        super().__init__()
        self.processor = processor
        self.transform = transform
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])


        
        if transform is None:
            self.transform = A.Compose([
                # Resize the image to the desired size and keep their aspect ratio
                A.LongestMaxSize(max_size=(size, size)),

                # Padding
                A.PadIfNeeded(
                    min_height=size, 
                    min_width=size,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[114, 114, 114]
                ),

                # Augmentation
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ],
            # YOLO Annotation Format
            bbox_params=A.BboxParams(
                format='yolo', 
                label_fields=['class_labels'],
                min_visibility=0.1  # drop if box is too small
            ))

        else:
            self.transform = transform
            
    def __len__(self) -> int:
        return len(self.image_files)


    def __getitem__(self, index: int) -> torch.Tensor:
        # construct image path
        img_name = self.image_files[index]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Assume label file name is the same as image file name (ex: 1.jpg -> 1.txt)
        label_name = os.path.splitext(img_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        # read image
        image = Image.open(img_path).convert("RGB")
        
        # read label (YOLO format: class cx cy w h)
        boxes, class_labels = get_box(label_path)

        # apply transform
        transformed = self.transform(image = image, bboxes = boxes, class_labels = class_labels)
        pixel_values = transformed['image']


def get_box(label_path):
    """
    Get the boxes and class labels from the label file.
    
    Args:
        label_path: The path to the label file.
        
    Returns:
        boxes: A list of boxes in YOLO format.
        class_labels: A list of class labels.
    """

    boxes = []
    class_labels = []

    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = list(map(float, line.strip().split()))
                cls_id = int(parts[0])
                # box = [cx, cy, w, h]
                box = parts[1:] 
                
                # check if box is valid
                if len(box) == 4:
                    boxes.append(box)
                    class_labels.append(cls_id)

    return boxes, class_labels

