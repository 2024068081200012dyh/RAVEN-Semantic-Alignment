import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import os

class RAVENDataset(Dataset):
    """
    Custom Dataset loader for MS COCO and LVIS v1.0.
    Supports on-the-fly generation of semantic masks for TDSR (Innovation 3).
    """
    def __init__(self, root, ann_file, img_size=640, is_train=True):
        self.root = root
        self.img_size = img_size
        self.is_train = is_train
        # In a real scenario, you would load COCO/LVIS API here
        # self.coco = COCO(ann_file)
        self.ids = list(range(100))  # Mock data indices

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Returns:
            image: Processed image tensor.
            targets: Dictionary containing bboxes, labels, and small_region_masks.
        """
        # Load mock image (replace with actual loading logic)
        image = torch.randn(3, self.img_size, self.img_size)
        
        # Mock ground truth: [x1, y1, x2, y2]
        bboxes = torch.tensor([[10, 10, 50, 50], [100, 100, 400, 400]], dtype=torch.float32)
        labels = torch.tensor([1, 5], dtype=torch.int64)
        
        # Generate Small Region Mask for TDSR
        # Objects with area < 32*32 are marked for targeted distillation
        small_region_mask = self._generate_small_mask(bboxes)

        targets = {
            "boxes": bboxes,
            "labels": labels,
            "small_mask": small_region_mask
        }

        return image, targets

    def _generate_small_mask(self, bboxes):
        """
        Generates a spatial importance mask identifying small-scale objects.
        This directly supports the semantic recovery mechanism in TDSR.
        """
        mask = torch.zeros((1, self.img_size // 16, self.img_size // 16))
        for box in bboxes:
            area = (box[2] - box[0]) * (box[3] - box[1])
            if area < 1024:  # 32x32 threshold
                # Map image coordinates to feature map coordinates (stride 16)
                x1, y1, x2, y2 = (box / 16).int()
                mask[:, y1:y2, x1:x2] = 1.0
        return mask