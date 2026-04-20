import torch
import torch.nn.functional as F

def compute_tdsr_loss(s_feat, t_feat, bboxes, small_obj_threshold=1024):
    """
    Innovation 3: Token-wise Distillation on Small Regions (TDSR).
    Recovers semantic density for small-scale objects through localized knowledge transfer.
    
    Args:
        s_feat: Student feature maps [B, C, H, W]
        t_feat: Teacher feature maps [B, C, H, W]
        bboxes: List of ground truth bounding boxes [N, 4] (x1, y1, x2, y2)
        small_obj_threshold: Area threshold to define small objects (default: 32x32)
    """
    total_loss = 0.0
    batch_size = s_feat.size(0)
    h, w = s_feat.shape[-2:]

    for i in range(batch_size):
        # Calculate object areas
        current_boxes = bboxes[i]
        widths = current_boxes[:, 2] - current_boxes[:, 0]
        heights = current_boxes[:, 3] - current_boxes[:, 1]
        areas = widths * heights
        
        # Filter indices for small objects
        small_mask = areas < small_obj_threshold
        target_boxes = current_boxes[small_mask]
        
        if target_boxes.size(0) == 0:
            continue
            
        # Create spatial mask for small regions
        spatial_mask = torch.zeros((1, 1, h, w), device=s_feat.device)
        for box in target_boxes:
            # Scale boxes to feature map size
            x1, y1 = int(box[0] * w), int(box[1] * h)
            x2, y2 = int(box[2] * w), int(box[3] * h)
            spatial_mask[:, :, y1:y2, x1:x2] = 1.0
            
        # Apply Token-wise Distillation
        diff = F.mse_loss(s_feat[i:i+1], t_feat[i:i+1], reduction='none')
        distill_loss = (diff * spatial_mask).sum() / (spatial_mask.sum() + 1e-6)
        total_loss += distill_loss
        
    return total_loss / batch_size