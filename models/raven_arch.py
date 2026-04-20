import torch
import torch.nn as nn
import torch.nn.functional as F

class LPAPAN(nn.Module):
    """
    Innovation 1: Language-aware Path Aggregation Network (LPA-PAN).
    Injects linguistic priors into the multi-scale feature fusion hierarchy to 
    enhance semantic-visual correspondence.
    """
    def __init__(self, in_channels, text_dim=512, expansion=0.5):
        super(LPAPAN, self).__init__()
        self.in_channels = in_channels
        self.inter_channels = int(in_channels * expansion)
        
        # Textual feature projection to visual space
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, self.inter_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.inter_channels, in_channels)
        )
        
        # Spatial and Channel-wise Attention Gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        
        # Final refinement layer
        self.refine = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)

    def forward(self, visual_feat, text_embeddings):
        """
        Args:
            visual_feat: [Batch, Channels, Height, Width]
            text_embeddings: [Batch, Num_Classes, Text_Dim]
        """
        # Aggregate global text context
        text_context = torch.mean(text_embeddings, dim=1)  # [B, D]
        projected_text = self.text_encoder(text_context).view(-1, self.in_channels, 1, 1)
        
        # Apply Language-aware Gating
        attention_mask = self.gate_conv(projected_text.expand_as(visual_feat))
        fused_feat = visual_feat * attention_mask
        
        return self.refine(fused_feat) + visual_feat

class PromptAnchor(nn.Module):
    """
    Innovation 2: Learnable Prompt Anchors.
    A scale-conditioned prompting mechanism to bridge regional visual features 
    and textual embeddings.
    """
    def __init__(self, num_anchors=100, embed_dim=256):
        super(PromptAnchor, self).__init__()
        self.num_anchors = num_anchors
        
        # Learnable regional anchors
        self.anchor_embeddings = nn.Parameter(torch.randn(num_anchors, embed_dim))
        nn.init.normal_(self.anchor_embeddings, std=0.02)
        
        # Scale-conditioned adapter
        self.adapter = nn.Sequential(
            nn.Conv2d(embed_dim, num_anchors, kernel_size=1),
            nn.LayerNorm([num_anchors, 1, 1], elementwise_affine=False),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        """
        Generates region-aware alignment maps.
        """
        # Calculate similarity response maps
        response_maps = self.adapter(x)  # [B, K, H, W]
        return response_maps

class RAVEN(nn.Module):
    def __init__(self, num_classes=1203, embed_dim=256):
        super(RAVEN, self).__init__()
        self.neck = LPAPAN(in_channels=embed_dim)
        self.prompt_engine = PromptAnchor(num_anchors=128, embed_dim=embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x, text_embeddings):
        # Feature enhancement via LPA-PAN
        enhanced_feat = self.neck(x, text_embeddings)
        
        # Region-aware alignment via Prompt Anchors
        response_maps = self.prompt_engine(enhanced_feat)
        
        # Global pooling for classification
        pooled_feat = torch.mean(enhanced_feat, dim=[2, 3])
        logits = self.classifier(pooled_feat)
        
        return logits, response_maps, enhanced_feat