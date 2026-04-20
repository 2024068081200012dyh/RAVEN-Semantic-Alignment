import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

class RAVENVisualizer:
    """
    Utility class to generate Similarity Response Maps and Feature Activations.
    This addresses the editor's request to enhance the persuasiveness of 
    regional alignment (as shown in Figure 5 of the manuscript).
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device

    def preprocess(self, image_path, img_size=640):
        """Standard preprocessing for input images."""
        raw_image = Image.open(image_path).convert('RGB')
        raw_image = raw_image.resize((img_size, img_size))
        image_tensor = np.array(raw_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_tensor).permute(2, 0, 1).unsqueeze(0)
        return image_tensor.to(self.device), raw_image

    def generate_heatmap(self, response_map, original_image, alpha=0.6, colormap=cv2.COLORMAP_JET):
        """
        Overlays the regional response map onto the original image.
        Args:
            response_map: Tensor of shape [H, W] (similarity scores)
            original_image: PIL Image
        """
        # Normalize response map to [0, 255]
        res_map = response_map.cpu().detach().numpy()
        res_map = (res_map - res_map.min()) / (res_map.max() - res_map.min() + 1e-8)
        res_map = (res_map * 255).astype(np.uint8)

        # Resize heatmap to match original image size
        img_array = np.array(original_image)
        heatmap = cv2.resize(res_map, (img_array.shape[1], img_array.shape[0]))
        heatmap = cv2.applyColorMap(heatmap, colormap)

        # Superimpose the heatmap on original image
        combined_img = cv2.addWeighted(img_array, 1 - alpha, heatmap, alpha, 0)
        return combined_img

    def plot_and_save(self, image_path, text_query, save_path="results/vis_alignment.png"):
        """
        Main execution: takes an image and a text query, saves the alignment visualization.
        """
        img_tensor, raw_img = self.preprocess(image_path)
        
        # Mock text embedding for the specific query (e.g., 'cat' or 'remote controller')
        # In actual use, this would come from a CLIP-based text encoder
        text_embedding = torch.randn(1, 1, 512).to(self.device)

        with torch.no_grad():
            # Extract response maps from the Prompt Anchor module
            _, resp_maps, _ = self.model(img_tensor, text_embedding)
        
        # Take the first anchor's response map for visualization
        # Shape: [B, K, H, W] -> Pick one from K anchors
        target_map = resp_maps[0, 0] 
        
        vis_result = self.generate_heatmap(target_map, raw_img)
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_img)
        plt.title(f"Input: '{text_query}'")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(vis_result, cv2.COLOR_BGR2RGB))
        plt.title("RAVEN Region Alignment")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        print(f"Success: Visualization saved to {save_path}")

if __name__ == "__main__":
    # Example usage for testing the script
    from models.raven_arch import RAVEN
    
    # Initialize model with placeholder weights
    model_instance = RAVEN()
    visualizer = RAVENVisualizer(model_instance, device='cpu') # Use CPU for demo
    
    # Run visualization on a sample image
    # Note: Ensure you have a 'sample.jpg' in data/ or provide a valid path
    try:
        visualizer.plot_and_save("data/sample.jpg", "Target Object")
    except Exception as e:
        print(f"Note: To fully run this, please provide a valid 'data/sample.jpg'. Error: {e}")