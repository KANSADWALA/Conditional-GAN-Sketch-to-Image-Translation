import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import numpy as np
from PIL import Image
import os
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
warnings.filterwarnings('ignore', category=UserWarning, message='.*Glyph.*')

# ==================== CONFIGURATION - EDIT THESE ====================
MODE = 'test_single'  # Options: 'train', 'test_single', 'test_batch'

# Training settings
NUM_EPOCHS = 200
BATCH_SIZE = 4
NUM_SYNTHETIC_SAMPLES = 300  # Number of synthetic images to generate

# Testing settings
SINGLE_IMAGE_PATH = 'datasets/synthetic/test/img_0020.png'  # For test_single mode
IMAGE_FOLDER = 'path/to/your/images/'  # For test_batch mode
MODEL_PATH = 'outputs/checkpoints/model_epoch_100.pth'  # Model to use for testing

# ==================== Dataset Generator ====================
class SyntheticDatasetGenerator:
    """Generate synthetic dataset for image-to-image translation"""

    @staticmethod
    def create_synthetic_dataset(num_samples=200, save_dir='datasets/synthetic'):
        """
        Create a synthetic dataset for testing
        - Input: Simple shapes/sketches
        - Output: Colored versions
        """
        print("=" * 70)
        print("CREATING SYNTHETIC DATASET")
        print("=" * 70)
        print(f"Generating {num_samples} synthetic image pairs...")
        print("Input: Sketch/edge representations")
        print("Output: Colored versions")
        print("=" * 70)

        os.makedirs(f'{save_dir}/train', exist_ok=True)
        os.makedirs(f'{save_dir}/test', exist_ok=True)

        train_samples = int(num_samples * 0.8)
        test_samples = num_samples - train_samples

        for split, n_samples in [('train', train_samples), ('test', test_samples)]:
            split_dir = f'{save_dir}/{split}'

            for i in tqdm(range(n_samples), desc=f'Creating {split} set'):
                # Create input (sketch/edge)
                img_size = 256
                input_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
                target_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

                # Draw random shapes (simulating building elements)
                num_shapes = np.random.randint(3, 8)
                for _ in range(num_shapes):
                    shape_type = np.random.choice(['rectangle', 'window_grid', 'circle'])
                    x, y = np.random.randint(20, img_size-100, 2)

                    if shape_type == 'rectangle':
                        w, h = np.random.randint(40, 120, 2)
                        # Input: Black outline
                        input_img[y:y+h, x:x+w] = 0
                        input_img[y+2:y+h-2, x+2:x+w-2] = 255  # Hollow
                        # Target: Colored fill
                        color = np.random.randint(100, 200, 3)
                        target_img[y:y+h, x:x+w] = color

                    elif shape_type == 'window_grid':
                        # Window grid
                        w, h = np.random.randint(60, 100, 2)
                        grid_size = 10
                        for gx in range(0, w, grid_size):
                            for gy in range(0, h, grid_size):
                                if x+gx < img_size and y+gy < img_size:
                                    input_img[y+gy:y+gy+grid_size-2, x+gx:x+gx+grid_size-2] = 0
                                    target_img[y+gy:y+gy+grid_size-2, x+gx:x+gx+grid_size-2] = [50, 100, 200]

                    else:  # circle
                        radius = np.random.randint(20, 50)
                        cy, cx = y + radius, x + radius
                        for dy in range(-radius, radius):
                            for dx in range(-radius, radius):
                                if dx*dx + dy*dy <= radius*radius:
                                    ny, nx = cy + dy, cx + dx
                                    if 0 <= ny < img_size and 0 <= nx < img_size:
                                        if dx*dx + dy*dy > (radius-3)*(radius-3):
                                            input_img[ny, nx] = 0
                                        color = np.random.randint(80, 180, 3)
                                        target_img[ny, nx] = color

                # Combine input and target side by side
                combined = np.concatenate([input_img, target_img], axis=1)
                Image.fromarray(combined).save(f'{split_dir}/img_{i:04d}.png')

        print(f"\n✓ Synthetic dataset created successfully!")
        print(f"✓ Location: {save_dir}")
        print(f"✓ Training samples: {train_samples}")
        print(f"✓ Test samples: {test_samples}")
        print("=" * 70)
        return save_dir

 
# ==================== Paired Image Dataset ====================
class PairedImageDataset(Dataset):
    """
    Dataset that loads paired images from combined images (left-right split)
    Compatible with pix2pix dataset format
    """
    def __init__(self, data_dir, mode='train', img_size=256):
        self.mode = mode
        self.img_size = img_size

        # Single directory with combined images (input | target)
        self.image_dir = os.path.join(data_dir, mode)

        if not os.path.exists(self.image_dir):
            raise ValueError(f"Directory not found: {self.image_dir}")

        self.files = sorted([f for f in os.listdir(self.image_dir)
                           if f.endswith(('.png', '.jpg', '.jpeg'))])

        if len(self.files) == 0:
            raise ValueError(f"No images found in {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print(f"✓ Loaded {len(self.files)} images from {self.image_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load combined image and split
        img_path = os.path.join(self.image_dir, self.files[idx])
        combined = Image.open(img_path).convert('RGB')
        w, h = combined.size

        # Split: left half = input (labels), right half = target (real image)
        input_img = combined.crop((0, 0, w//2, h))
        target_img = combined.crop((w//2, 0, w, h))

        input_tensor = self.transform(input_img)
        target_tensor = self.transform(target_img)

        return input_tensor, target_tensor


# ==================== Spectral Normalization ====================
def spectral_norm(module, name='weight', n_power_iterations=1, eps=1e-12):
    """Apply spectral normalization to conv layers"""
    return nn.utils.spectral_norm(module, name, n_power_iterations, eps)


# ==================== Generator (U-Net) ====================
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU()
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()

        # Encoder
        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # Decoder
        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x, return_bottleneck=False):
        # Encoder with skip connections
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)  # Bottleneck

        # Add this debug line temporarily:
        #print(f"DEBUG: d8 shape = {d8.shape}")  # Should be [batch, 512, 1, 1]

        # Decoder with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        output = self.final(u7)

        if return_bottleneck:
            return output, d8
        return output


# ==================== Discriminator (PatchGAN) ====================
class Discriminator(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1))]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv1 = nn.Sequential(*discriminator_block(in_channels, 64, normalize=False))
        self.conv2 = nn.Sequential(*discriminator_block(64, 128))
        self.conv3 = nn.Sequential(*discriminator_block(128, 256))
        self.conv4 = nn.Sequential(*discriminator_block(256, 512))
        self.final = nn.Sequential(
            nn.ZeroPad2d((1, 0, 1, 0)),
            spectral_norm(nn.Conv2d(512, 1, 4, padding=1))
        )

    def forward(self, img_input, img_target, return_features=False):
        x = torch.cat((img_input, img_target), 1)

        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        output = self.final(f4)

        if return_features:
            return output, [f1, f2, f3, f4]
        return output
    

def print_model_summary(model, model_name, input_size):
    """
    Print detailed model summary with layer information
    
    Args:
        model: PyTorch model
        model_name: Name to display
        input_size: Tuple of input dimensions (batch, channels, height, width)
    """
    print("\n" + "="*90)
    print(f"{model_name.upper()} ARCHITECTURE SUMMARY")
    print("="*90)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 PARAMETER STATISTICS:")
    print(f"   Total Parameters:      {total_params:,}")
    print(f"   Trainable Parameters:  {trainable_params:,}")
    print(f"   Non-trainable:         {total_params - trainable_params:,}")
    print(f"   Model Size:            {total_params * 4 / (1024**2):.2f} MB (float32)")
    
    print(f"\n🏗️  LAYER-BY-LAYER BREAKDOWN:")
    print("-"*90)
    print(f"{'Layer Name':<40} {'Output Shape':<25} {'Params':<15}")
    print("-"*90)
    
    # Create dummy input
    device = next(model.parameters()).device
    if model_name == "Generator":
        x = torch.randn(input_size).to(device)
    else:  # Discriminator
        x1 = torch.randn(input_size).to(device)
        x2 = torch.randn(input_size).to(device)
    
    # Hook to capture layer outputs
    layer_outputs = []
    hooks = []
    
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            layer_outputs.append((module.__class__.__name__, list(output.shape)))
    
    # Register hooks
    for name, layer in model.named_modules():
        if len(list(layer.children())) == 0:  # Only leaf modules
            hooks.append(layer.register_forward_hook(hook_fn))
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        if model_name == "Generator":
            _ = model(x)
        else:  # Discriminator
            _ = model(x1, x2)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Print layers with their outputs
    current_params = 0
    for i, (layer_name, layer) in enumerate(model.named_modules()):
        if len(list(layer.children())) == 0:  # Only leaf modules
            # Count parameters in this layer
            layer_params = sum(p.numel() for p in layer.parameters())
            current_params += layer_params
            
            # Get output shape if available
            if i < len(layer_outputs):
                output_shape = str(layer_outputs[i][1])
            else:
                output_shape = "N/A"
            
            # Format layer name
            display_name = f"{layer.__class__.__name__}"
            if layer_name:
                display_name = f"{layer_name} ({layer.__class__.__name__})"
            
            # Print row
            if layer_params > 0:
                print(f"{display_name:<40} {output_shape:<25} {layer_params:>12,}")
    
    print("-"*90)
    print(f"{'TOTAL':<40} {'':<25} {total_params:>12,}")
    print("-"*90)
    
    # Architecture visualization
    print(f"\n🔄 INFORMATION FLOW:")
    if model_name == "Generator":
        print(f"   Input:  {input_size}")
        print(f"   ↓ Encoder (8 down-sampling layers)")
        print(f"   ↓ Bottleneck: [batch, 512, 1, 1]")
        print(f"   ↓ Decoder (7 up-sampling layers with skip connections)")
        print(f"   Output: [batch, 3, 256, 256]")
    else:  # Discriminator
        print(f"   Input A: {input_size}")
        print(f"   Input B: {input_size}")
        print(f"   ↓ Concatenate: [batch, 6, 256, 256]")
        print(f"   ↓ 4 Convolutional blocks with Spectral Norm")
        print(f"   Output: PatchGAN predictions [batch, 1, 30, 30]")
    
    print("="*90 + "\n")
    
    model.train()


def print_complete_model_summary(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Print summary for both Generator and Discriminator
    """
    print("\n" + "🚀 "*30)
    print("GAN MODEL ARCHITECTURE ANALYSIS")
    print("🚀 "*30)
    
    # Initialize models
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    # Print summaries
    print_model_summary(generator, "Generator", (1, 3, 256, 256))
    print_model_summary(discriminator, "Discriminator", (1, 3, 256, 256))
    
    # Combined statistics
    total_gen_params = sum(p.numel() for p in generator.parameters())
    total_disc_params = sum(p.numel() for p in discriminator.parameters())
    total_combined = total_gen_params + total_disc_params
    
    print("\n" + "="*90)
    print("📈 COMBINED MODEL STATISTICS")
    print("="*90)
    print(f"   Generator Parameters:     {total_gen_params:>15,}")
    print(f"   Discriminator Parameters: {total_disc_params:>15,}")
    print(f"   {'─'*40}")
    print(f"   Total GAN Parameters:     {total_combined:>15,}")
    print(f"   Total Model Size:         {total_combined * 4 / (1024**2):>14.2f} MB")
    print("="*90)
    
    # Memory estimation
    print(f"\n💾 MEMORY ESTIMATION (Training):")
    batch_size = 4
    memory_per_param = 4  # bytes (float32)
    optimizer_multiplier = 3  # Adam needs ~3x memory (params + gradients + momentum)
    
    model_memory = total_combined * memory_per_param / (1024**2)
    optimizer_memory = model_memory * optimizer_multiplier
    activation_memory = batch_size * (256 * 256 * 3) * 4 * 10 / (1024**2)  # Rough estimate
    
    total_memory = model_memory + optimizer_memory + activation_memory
    
    print(f"   Model Weights:        {model_memory:>8.2f} MB")
    print(f"   Optimizer States:     {optimizer_memory:>8.2f} MB")
    print(f"   Activations:          {activation_memory:>8.2f} MB (batch_size={batch_size})")
    print(f"   {'─'*40}")
    print(f"   Estimated Total:      {total_memory:>8.2f} MB")
    print(f"   Recommended VRAM:     {total_memory * 1.5:>8.2f} MB (with safety margin)")
    print("="*90 + "\n")


# ==================== CORRECTED Latent Visualization with Labels ====================
def visualize_latent(latent, ax_map, ax_channels):
    """
    Create more informative and varied latent space visualizations
    Shows different patterns for different inputs - WITH LABELS
    Handles edge cases where latent has fewer than 3 channels
    """
    latent_cpu = latent.cpu()
    
    # Handle both 3D and 4D tensors
    if latent_cpu.dim() == 3:
        latent_cpu = latent_cpu.unsqueeze(0)  # Add batch dimension
    
    batch_size, num_channels, height, width = latent_cpu.shape
    
    print(f"\n🔍 Latent tensor debug:")
    print(f"   Shape: {latent_cpu.shape}")
    print(f"   Min: {latent_cpu.min().item():.4f}, Max: {latent_cpu.max().item():.4f}")
    print(f"   Mean: {latent_cpu.mean().item():.4f}, Std: {(latent_cpu.std().item() if num_channels > 1 else 0.0):.4f}")
    
    # ===== HANDLE EDGE CASE: Less than 3 channels =====
    if num_channels < 3:
        print(f"   ⚠️  Warning: Only {num_channels} channel(s) available. Using grayscale visualization.")
        
        # Show single channel as grayscale
        channel_data = latent_cpu[0, 0].numpy()
        p_low, p_high = np.percentile(channel_data, [2, 98])
        if p_high != p_low:
            channel_norm = np.clip((channel_data - p_low) / (p_high - p_low), 0, 1)
        else:
            channel_norm = np.zeros_like(channel_data)
        
        ax_map.imshow(channel_norm, cmap='viridis', interpolation='bilinear')
        ax_map.set_title(f'🧠 Latent Space\n(Single Channel - Ch0)',
                         fontsize=11, fontweight='bold', color='#9b59b6')
        ax_map.axis('off')
        
        # Simple bar chart for channel activation
        ax_channels.clear()
        ax_channels.bar([0], [channel_norm.mean()], color='#8e44ad', alpha=0.7)
        ax_channels.set_title('⚡ Channel Activation', fontsize=11, fontweight='bold', color='#9b59b6')
        ax_channels.set_xlabel('Channel', fontweight='bold')
        ax_channels.set_ylabel('Mean Activation', fontweight='bold')
        ax_channels.set_xticks([0])
        ax_channels.set_xticklabels(['Ch0'])
        ax_channels.grid(True, alpha=0.3)
        return
    
    # ===== NORMAL CASE: 3+ channels =====
    # Calculate channel importance using L2 norm
    channel_importance = []
    for c in range(num_channels):
        channel_data = latent_cpu[0, c]
        importance = torch.sqrt((channel_data ** 2).mean()).item()
        channel_importance.append(importance)
    
    # Get top 3 most important channels for RGB composite
    top_channels_idx = np.argsort(channel_importance)[-3:]
    
    print(f"   Top 3 channels: {top_channels_idx} with importance: {[channel_importance[i] for i in top_channels_idx]}")
    
    # Create RGB composite
    rgb_composite = np.zeros((height, width, 3))
    for i, channel_idx in enumerate(top_channels_idx):
        channel_data = latent_cpu[0, channel_idx].numpy()
        # Normalize to [0, 1] using percentile for better contrast
        p_low, p_high = np.percentile(channel_data, [2, 98])
        if p_high != p_low:
            channel_norm = np.clip((channel_data - p_low) / (p_high - p_low), 0, 1)
        else:
            channel_norm = np.zeros_like(channel_data)
        rgb_composite[:, :, i] = channel_norm
    
    # Display
    ax_map.imshow(rgb_composite, interpolation='bilinear')
    ax_map.set_title(f'🧠 Latent Space\nTop Features: Ch{top_channels_idx[0]},{top_channels_idx[1]},{top_channels_idx[2]}',
                     fontsize=11, fontweight='bold', color='#9b59b6')
    ax_map.axis('off')
    
    # Add RGB channel labels on the image
    ax_map.text(0.05, 0.95, 'R', transform=ax_map.transAxes, fontsize=14, 
               fontweight='bold', color='red', va='top',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='red', linewidth=2))
    ax_map.text(0.05, 0.80, f'Ch{top_channels_idx[0]}', transform=ax_map.transAxes, 
               fontsize=9, color='red', va='top', fontweight='bold')
    
    ax_map.text(0.50, 0.95, 'G', transform=ax_map.transAxes, fontsize=14, 
               fontweight='bold', color='green', va='top', ha='center',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='green', linewidth=2))
    ax_map.text(0.50, 0.80, f'Ch{top_channels_idx[1]}', transform=ax_map.transAxes, 
               fontsize=9, color='green', va='top', ha='center', fontweight='bold')
    
    ax_map.text(0.95, 0.95, 'B', transform=ax_map.transAxes, fontsize=14, 
               fontweight='bold', color='blue', va='top', ha='right',
               bbox=dict(boxstyle='circle,pad=0.3', facecolor='white', edgecolor='blue', linewidth=2))
    ax_map.text(0.95, 0.80, f'Ch{top_channels_idx[2]}', transform=ax_map.transAxes, 
               fontsize=9, color='blue', va='top', ha='right', fontweight='bold')
    
    # ==== CHANNEL ACTIVATIONS: Show MULTIPLE statistics ====
    channel_l2_norms = []
    channel_means = []
    channel_max_activations = []
    
    for c in range(num_channels):
        channel_data = latent_cpu[0, c]
        channel_l2_norms.append(torch.sqrt((channel_data ** 2).mean()).item())
        channel_means.append(channel_data.mean().item())
        channel_max_activations.append(channel_data.abs().max().item())
    
    channel_l2_norms = np.array(channel_l2_norms)
    channel_means = np.array(channel_means)
    channel_max_activations = np.array(channel_max_activations)
    
    print(f"   L2 norms - Min: {channel_l2_norms.min():.4f}, Max: {channel_l2_norms.max():.4f}")
    print(f"   Means - Min: {channel_means.min():.4f}, Max: {channel_means.max():.4f}")
    
    # Downsample if too many channels
    stride = max(1, len(channel_l2_norms) // 50)
    x_indices = np.arange(len(channel_l2_norms))
    x_plot = x_indices[::stride]
    
    norms_plot = channel_l2_norms[::stride]
    means_plot = channel_means[::stride]
    max_plot = channel_max_activations[::stride]
    
    # Create twin axis for better visualization
    ax_channels.clear()
    
    # Plot L2 norms (primary metric)
    color1 = '#8e44ad'
    line1, = ax_channels.plot(x_plot, norms_plot, color=color1, linewidth=2.5,
                    marker='o', markersize=3, alpha=0.8, label='Activation Strength (L2)')
    ax_channels.fill_between(x_plot, 0, norms_plot, alpha=0.3, color=color1)
    
    # Highlight top activated channels with stars AND labels (IMPROVED - no overlap)
    label_positions = []  # Track label positions to avoid overlap

    for i, idx in enumerate(top_channels_idx):
        if idx % stride == 0:
            plot_idx = idx // stride
            if plot_idx < len(x_plot):
                y_val = norms_plot[plot_idx]
                
                # Plot star marker
                ax_channels.plot(x_plot[plot_idx], y_val, 
                            marker='*', markersize=15, color='red', 
                            markeredgecolor='yellow', markeredgewidth=1.5,
                            zorder=5)
                
                # Smart label positioning to avoid overlap
                # Alternate positions: above, above-right, above-left
                if i == 0:
                    xytext = (0, 15)  # Center above
                elif i == 1:
                    xytext = (20, 15)  # Right
                else:
                    xytext = (-20, 15)  # Left
                
                # Add label with better positioning
                ax_channels.annotate(f'Ch{idx}\n{y_val:.3f}', 
                                xy=(x_plot[plot_idx], y_val),
                                xytext=xytext, textcoords='offset points',
                                ha='center', fontsize=7, fontweight='bold',
                                color='red',
                                bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='yellow', alpha=0.7, edgecolor='red', linewidth=1),
                                arrowprops=dict(arrowstyle='->', color='red', lw=1, alpha=0.6))
    
    ax_channels.set_ylabel('Activation Strength', fontweight='bold', fontsize=10, color=color1)
    ax_channels.tick_params(axis='y', labelcolor=color1)
    ax_channels.set_xlabel('Channel Index', fontweight='bold', fontsize=10)
    ax_channels.set_title('⚡ Channel Activation Strength\n(L2 Norm per Channel)',
                         fontsize=11, fontweight='bold', color='#9b59b6')
    ax_channels.grid(True, alpha=0.3, linestyle='--')
    
    # Set reasonable limits
    if len(x_plot) > 1:
        ax_channels.set_xlim([x_plot[0] - 1, x_plot[-1] + 1])
    if len(norms_plot) > 0 and norms_plot.max() > 0:
        ax_channels.set_ylim([0, norms_plot.max() * 1.4])  # More space for labels (was 1.25)
    
    # Add secondary axis for means
    ax2 = ax_channels.twinx()
    color2 = '#e67e22'
    line2, = ax2.plot(x_plot, means_plot, color=color2, linewidth=1.5,
            marker='s', markersize=2, alpha=0.6, linestyle='--', label='Mean Value')
    ax2.set_ylabel('Mean Value', fontweight='bold', fontsize=9, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.grid(False)
    
    # Add statistics
    active_channels = (channel_l2_norms > 0.1).sum()
    max_activation = channel_l2_norms.max()
    mean_activation = channel_l2_norms.mean()
    
    info_text = f"Active: {active_channels}/{num_channels}\nMax: {max_activation:.3f}\nAvg: {mean_activation:.3f}"
    ax_channels.text(0.02, 0.75, info_text, transform=ax_channels.transAxes,  # Changed from 0.98 to 0.75
                    ha='left', va='top', fontsize=8, family='monospace',  # Smaller font
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0e6ff', 
                            edgecolor='#9b59b6', linewidth=1.5, alpha=0.85))
    
    # Combine legends
    ax_channels.legend([line1, line2], ['Activation Strength (L2)', 'Mean Value'], 
                      loc='upper right', fontsize=8, framealpha=0.9)

# ==================== Visualization Tools ====================
class GANVisualizer:
    """Visualize GAN training process"""

    @staticmethod
    def denormalize(tensor):
        """Convert normalized tensor back to [0, 1] range"""
        return tensor * 0.5 + 0.5

    @staticmethod
    def visualize_training_step(input_img, real_img, fake_img, d_real, d_fake,
                               latent=None, save_path='visualization.png'):
        """Visualize a complete training step with enhanced graphics"""
        try:
            # Set style for better looking plots
            plt.style.use('seaborn-v0_8-darkgrid')
            fig = plt.figure(figsize=(24, 14), facecolor='#f8f9fa')

            # Add main title with styling
            fig.suptitle('GAN Training Visualization Dashboard',
                    fontsize=22, fontweight='bold', y=0.96, color='#2c3e50')  # Changed y from 0.98 to 0.96

            # Denormalize images
            input_img = GANVisualizer.denormalize(input_img.cpu())
            real_img = GANVisualizer.denormalize(real_img.cpu())
            fake_img = GANVisualizer.denormalize(fake_img.cpu())

            # Row 1: Main Image Comparison (larger subplots)
            gs = fig.add_gridspec(3, 6, hspace=0.45, wspace=0.50,  # Changed from 0.35 to 0.50
                     left=0.05, right=0.95, top=0.90, bottom=0.08)

            # Input Image
            ax1 = fig.add_subplot(gs[0, 0:2])
            ax1.imshow(input_img.permute(1, 2, 0))
            ax1.set_title('INPUT\n(Sketch/Edges)', fontsize=14, fontweight='bold',
                         pad=10, color='#2980b9')
            ax1.axis('off')
            ax1.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax1.transAxes,
                                       fill=False, edgecolor='#2980b9', linewidth=3))

            # Real Image
            ax2 = fig.add_subplot(gs[0, 2:4])
            ax2.imshow(real_img.permute(1, 2, 0))
            ax2.set_title('TARGET (Real)\n(Ground Truth)', fontsize=14, fontweight='bold',
                         pad=10, color='#27ae60')
            ax2.axis('off')
            ax2.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax2.transAxes,
                                       fill=False, edgecolor='#27ae60', linewidth=3))

            # Generated Image
            ax3 = fig.add_subplot(gs[0, 4:6])
            ax3.imshow(fake_img.permute(1, 2, 0))
            ax3.set_title('GENERATED (Fake)\n(AI Output)', fontsize=14, fontweight='bold',
                         pad=10, color='#e74c3c')
            ax3.axis('off')
            ax3.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax3.transAxes,
                                       fill=False, edgecolor='#e74c3c', linewidth=3))

            # Row 2: Analysis Visualizations

            # Pixel-wise difference heatmap
            ax4 = fig.add_subplot(gs[1, 0:2])
            diff = torch.abs(real_img - fake_img).mean(dim=0)
            im1 = ax4.imshow(diff, cmap='YlOrRd', interpolation='bilinear')
            ax4.set_title('🔥 Pixel-Wise Error Map\n(Lower is Better)',
                         fontsize=13, fontweight='bold', color='#d35400')
            ax4.axis('off')
            cbar1 = plt.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)
            cbar1.set_label('Error Magnitude', rotation=270, labelpad=20, fontsize=10)
            avg_error = diff.mean().item()
            ax4.text(0.5, -0.15, f'Avg Error: {avg_error:.4f}',
                    transform=ax4.transAxes, ha='center', fontsize=11,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ffe6e6', alpha=0.8))

            # Side-by-side comparison with split line
            ax5 = fig.add_subplot(gs[1, 2:4])
            comparison = torch.cat([real_img, fake_img], dim=2)
            ax5.imshow(comparison.permute(1, 2, 0))
            # Get actual image width and center the line
            img_width = comparison.shape[2]  # Width of concatenated image
            ax5.axvline(x=img_width/2, color='yellow', linewidth=3, linestyle='--', alpha=0.8)
            ax5.set_title('🔀 Side-by-Side Comparison\n(Real | Generated)',
                         fontsize=13, fontweight='bold', color='#8e44ad')
            ax5.axis('off')
            # Center labels based on actual image dimensions
            img_width = comparison.shape[2]
            ax5.text(img_width/4, 240, 'REAL', ha='center', fontsize=11, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round', facecolor='#27ae60', alpha=0.9))
            ax5.text(3*img_width/4, 240, 'FAKE', ha='center', fontsize=11, fontweight='bold',
                    color='white', bbox=dict(boxstyle='round', facecolor='#e74c3c', alpha=0.9))

            # RGB Channel Analysis
            ax6 = fig.add_subplot(gs[1, 4:6])
            real_rgb = real_img.mean(dim=[1, 2]).numpy()
            fake_rgb = fake_img.mean(dim=[1, 2]).numpy()
            x_pos = np.arange(3)
            width = 0.35
            colors_real = ['#e74c3c', '#27ae60', '#3498db']
            colors_fake = ['#c0392b', '#229954', '#2874a6']

            bars1 = ax6.bar(x_pos - width/2, real_rgb, width, label='Real',
                           color=colors_real, alpha=0.8, edgecolor='black', linewidth=1.5)
            bars2 = ax6.bar(x_pos + width/2, fake_rgb, width, label='Generated',
                           color=colors_fake, alpha=0.8, edgecolor='black', linewidth=1.5)

            ax6.set_title('🎨 RGB Channel Intensity\n(Color Distribution)',
                         fontsize=13, fontweight='bold', color='#16a085')
            ax6.set_xticks(x_pos)
            ax6.set_xticklabels(['Red', 'Green', 'Blue'], fontweight='bold')
            ax6.set_ylabel('Mean Intensity', fontweight='bold')
            ax6.legend(loc='upper right', framealpha=0.9, fontsize=10)
            ax6.grid(True, alpha=0.3, linestyle='--')
            ax6.set_ylim([0, max(real_rgb.max(), fake_rgb.max()) * 1.2])

            # Row 3: Discriminator Analysis
            d_real_map = d_real.cpu().squeeze()
            d_fake_map = d_fake.cpu().squeeze()

            # Discriminator heatmap for Real
            ax7 = fig.add_subplot(gs[2, 0])
            im2 = ax7.imshow(d_real_map, cmap='RdYlGn', vmin=-3, vmax=3, interpolation='bilinear')
            ax7.set_title(f'✅ D(Real)\nScore: {d_real_map.mean():.3f}',
                         fontsize=12, fontweight='bold', color='#27ae60')
            ax7.axis('off')
            cbar2 = plt.colorbar(im2, ax=ax7, fraction=0.046, pad=0.04)
            cbar2.ax.tick_params(labelsize=8)

            # Discriminator heatmap for Fake
            ax8 = fig.add_subplot(gs[2, 1])
            im3 = ax8.imshow(d_fake_map, cmap='RdYlGn', vmin=-3, vmax=3, interpolation='bilinear')
            ax8.set_title(f'❌ D(Fake)\nScore: {d_fake_map.mean():.3f}',
                         fontsize=12, fontweight='bold', color='#e74c3c')
            ax8.axis('off')
            cbar3 = plt.colorbar(im3, ax=ax8, fraction=0.046, pad=0.04)
            cbar3.ax.tick_params(labelsize=8)

            # Score distribution histogram
            ax9 = fig.add_subplot(gs[2, 2])
            ax9.hist(d_real_map.flatten().numpy(), bins=30, alpha=0.7,
                    label='Real', color='#27ae60', edgecolor='black', linewidth=1.2)
            ax9.hist(d_fake_map.flatten().numpy(), bins=30, alpha=0.7,
                    label='Fake', color='#e74c3c', edgecolor='black', linewidth=1.2)
            ax9.set_title('📊 Score Distribution\n(Discriminator Output)',
                         fontsize=12, fontweight='bold', color='#34495e')
            ax9.set_xlabel('Score Value', fontweight='bold', fontsize=10)
            ax9.set_ylabel('Frequency', fontweight='bold', fontsize=10)
            ax9.legend(loc='upper right', fontsize=10, framealpha=0.9)
            ax9.grid(True, alpha=0.3, linestyle='--')

            # Confidence gauge/metrics
            ax10 = fig.add_subplot(gs[2, 3])
            real_conf = torch.sigmoid(d_real_map.mean().detach()).item() * 100
            fake_conf = torch.sigmoid(d_fake_map.mean().detach()).item() * 100
            separation = d_real_map.mean() - d_fake_map.mean()

            # Create confidence bars
            metrics = ['Real\nConfidence', 'Fake as\nReal', 'Separation']
            values = [real_conf, fake_conf, separation * 10]  # Scale separation for visualization
            colors_metrics = ['#27ae60', '#e74c3c', '#3498db']

            bars = ax10.barh(metrics, [real_conf, fake_conf, abs(separation)*30],
                           color=colors_metrics, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax10.set_xlim([0, 100])
            ax10.set_title('📈 Quality Metrics\n(Percentage)',
                          fontsize=12, fontweight='bold', color='#8e44ad')
            ax10.set_xlabel('Confidence %', fontweight='bold', fontsize=10)
            ax10.grid(True, alpha=0.3, axis='x', linestyle='--')

            # Add value labels on bars
            for i, (bar, val) in enumerate(zip(bars, [real_conf, fake_conf, separation])):
                width = bar.get_width()
                label = f'{val:.1f}%' if i < 2 else f'{val:.3f}'
                ax10.text(width + 2, bar.get_y() + bar.get_height()/2, label,
                         ha='left', va='center', fontweight='bold', fontsize=10)

            # Latent space visualization (IMPROVED)
            if latent is not None:
                ax11 = fig.add_subplot(gs[2, 4])
                ax12 = fig.add_subplot(gs[2, 5])
                # Latent is already [512, 1, 1], just add batch dimension
                visualize_latent(latent.unsqueeze(0), ax11, ax12)  # Now [1, 512, 1, 1]

            # Add training quality indicator
            quality_score = (real_conf + (100 - fake_conf)) / 2
            if quality_score > 75:
                quality_text = "🌟 EXCELLENT"
                quality_color = '#27ae60'
            elif quality_score > 60:
                quality_text = "✅ GOOD"
                quality_color = '#f39c12'
            elif quality_score > 45:
                quality_text = "⚠️  FAIR"
                quality_color = '#e67e22'
            else:
                quality_text = "❌ POOR"
                quality_color = '#e74c3c'

            fig.text(0.5, 0.02, f'Training Quality: {quality_text} (Score: {quality_score:.1f}/100)',
                    ha='center', fontsize=13, fontweight='bold', color=quality_color,
                    bbox=dict(boxstyle='round,pad=0.7', facecolor='white',
                             edgecolor=quality_color, linewidth=2.5, alpha=0.95))

            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='#f8f9fa')
            plt.close() # Close plot to free memory
            print(f"✓ Enhanced visualization saved: {save_path}")

        except Exception as e:
            print(f"⚠️  Visualization error: {e}")
            import traceback
            traceback.print_exc()
            plt.close('all')


# ==================== Training ====================
class Pix2PixTrainer:
    def __init__(self, generator, discriminator, device='cuda', lr_g=0.0002,
                 lr_d=0.0001, lambda_l1=100.0):
        self.device = device
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)

        self.optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(0.5, 0.999))

        self.criterion_GAN = nn.MSELoss()
        self.criterion_L1 = nn.L1Loss()
        self.lambda_l1 = lambda_l1

        self.g_losses = []
        self.d_losses = []

    def train_step(self, real_input, real_target, visualize=False):
        batch_size = real_input.size(0)

        # Train Discriminator
        self.optimizer_D.zero_grad()

        if visualize:
            fake_target, latent = self.generator(real_input, return_bottleneck=True)
        else:
            fake_target = self.generator(real_input)
            latent = None

        pred_real = self.discriminator(real_input, real_target)
        loss_real = self.criterion_GAN(pred_real, torch.ones_like(pred_real) * 0.9)

        pred_fake = self.discriminator(real_input, fake_target.detach())
        loss_fake = self.criterion_GAN(pred_fake, torch.zeros_like(pred_fake))

        loss_D = (loss_real + loss_fake) * 0.5
        loss_D.backward()
        self.optimizer_D.step()

        # Train Generator
        self.optimizer_G.zero_grad()

        pred_fake = self.discriminator(real_input, fake_target)
        loss_GAN = self.criterion_GAN(pred_fake, torch.ones_like(pred_fake))
        loss_L1 = self.criterion_L1(fake_target, real_target) * self.lambda_l1

        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        self.optimizer_G.step()

        self.g_losses.append(loss_G.item())
        self.d_losses.append(loss_D.item())

        result = {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item(),
            'loss_GAN': loss_GAN.item(),
            'loss_L1': loss_L1.item()
        }

        if visualize:
            result['fake_target'] = fake_target
            result['pred_real'] = pred_real
            result['pred_fake'] = pred_fake
            result['latent'] = latent

        return result


# ==================== Main Training Function ====================
def train_gan(
    data_dir='datasets/synthetic',
    num_epochs=100,
    batch_size=4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_interval=10,
    visualize_every=50
):
    """Train GAN on synthetic dataset"""

    print("=" * 70)
    print("GAN IMAGE-TO-IMAGE TRANSLATION - SYNTHETIC DATASET")
    print("=" * 70)

    os.makedirs('outputs/checkpoints', exist_ok=True)
    os.makedirs('outputs/samples', exist_ok=True)
    os.makedirs('outputs/visualizations', exist_ok=True)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    trainer = Pix2PixTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        lr_g=0.0002,
        lr_d=0.0001,
        lambda_l1=100.0
    )

    # Load datasets
    print(f"\nLoading dataset from: {data_dir}")
    train_dataset = PairedImageDataset(data_dir, mode='train')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    try:
        test_dataset = PairedImageDataset(data_dir, mode='test')
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        has_test = True
    except:
        print("⚠️  No test set found, using training samples for validation")
        has_test = False

    print(f"\n✓ Device: {device}")
    print(f"✓ Training samples: {len(train_dataset)}")
    if has_test:
        print(f"✓ Test samples: {len(test_dataset)}")
    print(f"✓ Batch size: {batch_size}")
    print("=" * 70)

    # Training loop
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*70}")

        epoch_g_loss = 0
        epoch_d_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}')

        for i, (input_imgs, target_imgs) in enumerate(progress_bar):
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            should_visualize = (i == 0 and (epoch + 1) % 5 == 0)  # Visualize first batch every 5 epochs

            losses = trainer.train_step(input_imgs, target_imgs, visualize=should_visualize)

            epoch_g_loss += losses['loss_G']
            epoch_d_loss += losses['loss_D']

            progress_bar.set_postfix({
                'D': f"{losses['loss_D']:.4f}",
                'G': f"{losses['loss_G']:.4f}",
                'L1': f"{losses['loss_L1']:.4f}"
            })

            # Visualize
            if should_visualize:
                print(f"\n📊 Creating visualization...")
                with torch.no_grad():
                    fake_imgs, latent = generator(input_imgs, return_bottleneck=True)
                    pred_real = discriminator(input_imgs, target_imgs)
                    pred_fake = discriminator(input_imgs, fake_imgs)

                GANVisualizer.visualize_training_step(
                    input_imgs[0], target_imgs[0], fake_imgs[0],
                    pred_real[0], pred_fake[0], latent=latent[0],
                    save_path=f'outputs/visualizations/epoch_{epoch+1:03d}.png'
                )

        # Epoch summary
        print(f"\n{'='*70}")
        print(f"Epoch {epoch + 1}: G_loss={epoch_g_loss/len(train_loader):.4f}, "
              f"D_loss={epoch_d_loss/len(train_loader):.4f}")
        print(f"{'='*70}")

        # Save checkpoint
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = f'outputs/checkpoints/model_epoch_{epoch+1:03d}.pth'
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")

            # Save samples
            with torch.no_grad():
                generator.eval()
                sample_input = input_imgs[:4]
                sample_target = target_imgs[:4]
                sample_fake = generator(sample_input)

                comparison = torch.cat([
                    GANVisualizer.denormalize(sample_input),
                    GANVisualizer.denormalize(sample_target),
                    GANVisualizer.denormalize(sample_fake)
                ], dim=0)

                save_image(comparison, f'outputs/samples/epoch_{epoch+1:03d}.png', nrow=4)
                generator.train()

    print("\n✓ TRAINING COMPLETED!")
    return trainer


# ==================== Inference on Custom Images ====================
def test_on_custom_image(
    image_path,
    model_path='outputs/checkpoints/model_epoch_100.pth',
    output_dir='outputs/custom_results',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Test trained model on a single custom image

    Args:
        image_path: Path to input image (can be combined input|target or just input)
        model_path: Path to trained model checkpoint
        output_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    print("=" * 70)
    print("TESTING ON CUSTOM IMAGE")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first or provide a valid checkpoint path")
        return

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    checkpoint = torch.load(model_path, map_location=device)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])

    generator.eval()
    discriminator.eval()

    print(f"✓ Model loaded from: {model_path}")
    print(f"✓ Epoch: {checkpoint['epoch'] + 1}")

    # Load and process image
    if not os.path.exists(image_path):
        print(f"❌ Image not found at {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # Check if it's a combined image (input|target) or just input
    if w == 2 * h or w > h * 1.5:
        # Combined image - split it
        input_img = img.crop((0, 0, w//2, h))
        has_target = True
        target_img = img.crop((w//2, 0, w, h))
        print("✓ Detected combined image (input|target)")
    else:
        # Just input image
        input_img = img
        has_target = False
        target_img = None
        print("✓ Processing single input image")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    input_tensor = transform(input_img).unsqueeze(0).to(device)

    # Generate
    with torch.no_grad():
        fake_img, latent = generator(input_tensor, return_bottleneck=True)
        pred_fake = discriminator(input_tensor, fake_img)

    # Save results
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]

    # Save generated image
    output_path = os.path.join(output_dir, f'{name_without_ext}_generated.png')
    save_image(GANVisualizer.denormalize(fake_img), output_path)
    print(f"✓ Generated image saved: {output_path}")

    # Prepare target tensor if available
    target_tensor = None
    pred_real_for_viz = None # Initialize for visualization call
    if has_target:
        target_tensor = transform(target_img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_real_for_viz = discriminator(input_tensor, target_tensor)

    # Calculate pred_map and confidence for RESULTS section
    pred_map = pred_fake[0].cpu().squeeze()
    # Fix for UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.detach().clone()
    confidence = torch.sigmoid(pred_map.mean().detach().clone()).item() * 100

    # Use the same visualization function as training epochs
    viz_path = os.path.join(output_dir, f'{name_without_ext}_analysis.png')

    # If we have a target, use it; otherwise pass the input as fallback for visualization
    real_for_viz = target_tensor[0] if has_target else input_tensor[0]
    # Ensure pred_real_for_viz is set correctly, using pred_fake[0] if no target
    d_real_for_viz = pred_real_for_viz[0] if has_target and pred_real_for_viz is not None else pred_fake[0] 

    GANVisualizer.visualize_training_step(
        input_tensor[0],
        real_for_viz,
        fake_img[0],
        d_real_for_viz, # Use the correctly assigned variable
        pred_fake[0],
        latent=latent[0],
        save_path=viz_path
    )

    print(f"✓ Visualization saved: {viz_path}")

    print(f"\n{'='*70}")
    print("RESULTS:")
    print(f"  Discriminator Score: {pred_map.mean():.3f}")
    print(f"  Realism Confidence: {confidence:.1f}%")
    print(f"  Quality: {'High' if confidence > 70 else 'Moderate' if confidence > 50 else 'Low'}")
    print(f"{'='*70}\n")


def batch_test_custom_images(
    image_folder,
    model_path='outputs/checkpoints/model_epoch_100.pth',
    output_dir='outputs/custom_results',
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Test trained model on multiple custom images from a folder

    Args:
        image_folder: Folder containing input images
        model_path: Path to trained model checkpoint
        output_dir: Directory to save results
        device: 'cuda' or 'cpu'
    """
    print("=" * 70)
    print("BATCH TESTING ON CUSTOM IMAGES")
    print("=" * 70)

    if not os.path.exists(image_folder):
        print(f"❌ Folder not found: {image_folder}")
        return

    # Get all image files
    image_files = [f for f in os.listdir(image_folder)
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    if len(image_files) == 0:
        print(f"❌ No images found in {image_folder}")
        return

    print(f"✓ Found {len(image_files)} images")
    print(f"✓ Processing...\n")

    for img_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(image_folder, img_file)
        print(f"\n{'='*70}")
        print(f"Processing: {img_file}")
        print(f"{'='*70}")
        test_on_custom_image(image_path, model_path, output_dir, device)

    print(f"\n{'='*70}")
    print(f"✓ BATCH PROCESSING COMPLETED!")
    print(f"✓ All results saved to: {output_dir}")
    print(f"{'='*70}\n")


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":

    print("\n" + "="*70)
    print("  GAN IMAGE-TO-IMAGE TRANSLATION - SYNTHETIC DATASET")
    print("="*70)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {DEVICE}")
    print(f"📋 Mode: {MODE.upper()}")

    if MODE == 'train':
        # Model summar\'
        print_complete_model_summary(DEVICE)
        # Training mode
        print("\n" + "="*70)
        print("STEP 1: CHECKING/CREATING SYNTHETIC DATASET")
        print("="*70)

        dataset_path = 'datasets/synthetic'
        
        # Check if dataset already exists
        train_dir = os.path.join(dataset_path, 'train')
        test_dir = os.path.join(dataset_path, 'test')
        
        dataset_exists = (os.path.exists(train_dir) and 
                        os.path.exists(test_dir) and
                        len([f for f in os.listdir(train_dir) if f.endswith('.png')]) > 0 and
                        len([f for f in os.listdir(test_dir) if f.endswith('.png')]) > 0)
        
        if dataset_exists:
            train_count = len([f for f in os.listdir(train_dir) if f.endswith('.png')])
            test_count = len([f for f in os.listdir(test_dir) if f.endswith('.png')])
            print(f"\n✓ Dataset already exists!")
            print(f"✓ Location: {dataset_path}")
            print(f"✓ Training samples: {train_count}")
            print(f"✓ Test samples: {test_count}")
            print(f"✓ Skipping dataset generation...")
            print("="*70)
        else:
            print("\n📦 Dataset not found. Generating new synthetic dataset...")
            generator_tool = SyntheticDatasetGenerator()
            dataset_path = generator_tool.create_synthetic_dataset(
                num_samples=NUM_SYNTHETIC_SAMPLES,
                save_dir=dataset_path
            )

            if dataset_path is None:
                print("❌ Failed to create dataset")
                exit(1)

        print("\n" + "="*70)
        print("STEP 2: TRAINING GAN")
        print("="*70)

        trainer = train_gan(
            data_dir=dataset_path,
            num_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            device=DEVICE,
            save_interval=10,
            visualize_every=50
        )

        print("\n✓ Training completed!")

    elif MODE == 'test_single':
        # Test on single image
        if not os.path.exists(SINGLE_IMAGE_PATH):
            print(f"❌ Image not found at: {SINGLE_IMAGE_PATH}")
            print("Please update SINGLE_IMAGE_PATH in the configuration section")
            print("\nExample paths to try:")
            print("  SINGLE_IMAGE_PATH = 'my_image.jpg'")
            print("  SINGLE_IMAGE_PATH = '/content/my_image.jpg'")
            print("  SINGLE_IMAGE_PATH = 'datasets/synthetic/test/img_0001.png'")
            exit(1)

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            print("Please train the model first by setting MODE = 'train'")
            print("Or update MODEL_PATH to point to an existing checkpoint")
            exit(1)

        test_on_custom_image(
            image_path=SINGLE_IMAGE_PATH,
            model_path=MODEL_PATH,
            output_dir='outputs/custom_results',
            device=DEVICE
        )

    elif MODE == 'test_batch':
        # Test on multiple images
        if not os.path.exists(IMAGE_FOLDER):
            print(f"❌ Folder not found at: {IMAGE_FOLDER}")
            print("Please update IMAGE_FOLDER in the configuration section")
            print("\nExample paths to try:")
            print("  IMAGE_FOLDER = 'my_images/'")
            print("  IMAGE_FOLDER = '/content/my_images/'")
            print("  IMAGE_FOLDER = 'datasets/synthetic/test/'")
            exit(1)

        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model not found at: {MODEL_PATH}")
            print("Please train the model first by setting MODE = 'train'")
            print("Or update MODEL_PATH to point to an existing checkpoint")
            exit(1)

        batch_test_custom_images(
            image_folder=IMAGE_FOLDER,
            model_path=MODEL_PATH,
            output_dir='outputs/custom_results',
            device=DEVICE
        )

    else:
        print(f"❌ Invalid MODE: {MODE}")
        print("Valid options: 'train', 'test_single', 'test_batch'")
        exit(1)

    print("\n" + "="*70)
    print("✓ ALL OPERATIONS COMPLETED!")
    print("="*70)

    print("\n💡 USAGE GUIDE:")
    print("\n📝 Edit the configuration at the top of the file:")
    print("   MODE = 'train'  # or 'test_single' or 'test_batch'")
    print("   NUM_EPOCHS = 100")
    print("   BATCH_SIZE = 1")
    print("   NUM_SYNTHETIC_SAMPLES = 200  # Number of synthetic images to generate")
    print("   SINGLE_IMAGE_PATH = 'path/to/your/image.jpg'")
    print("   IMAGE_FOLDER = 'path/to/your/folder/'")
    print("   MODEL_PATH = 'outputs/checkpoints/model_epoch_100.pth'")

    print("\n📁 OUTPUT LOCATIONS:")
    print("├── datasets/synthetic/      # Generated synthetic dataset")
    print("├── outputs/checkpoints/     # Trained models")
    print("├── outputs/samples/         # Training samples")
    print("├── outputs/visualizations/  # Training visualizations")
    print("└── outputs/custom_results/  # Your custom image results")

    print("\n📊 SYNTHETIC DATASET INFO:")
    print("  • Input: Sketch/edge representations (shapes, grids, circles)")
    print("  • Output: Colored versions of the sketches")
    print("  • Perfect for testing and experimentation")
    print("  • No downloads required - everything runs locally")

    print("\n" + "="*70 + "\n")