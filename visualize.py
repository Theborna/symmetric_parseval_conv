import matplotlib.pyplot as plt
import torch.nn as nn
from parseval_cnn import *
import json
import os
import os.path as path
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import math
from layers.averaged import AveragedDenoiser
from layers.BCOP.core import LipschitzModuleL2

def visualize_kernels(model, num_kernels=6, save_dir='kernel_visualizations', print_lipschitz=False):
    # Create the directory to save the images if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Get all SymSVD layers from the model
    conv_layers = [layer for layer in model.network if isinstance(layer, LipschitzModuleL2)]
    
    for idx, conv_layer in tqdm(enumerate(conv_layers), total=len(conv_layers), desc=f"Plotting kernel for each convoloution layer"):
        # Extract the weight of the convolutional kernel
        conv_layer._gen_weights()
        kernels = conv_layer.weight.data.cpu().numpy()
        conv_layer._input_shape = kernels.shape[:2]
        if print_lipschitz:
            print(f"Layer {idx} Lipschitz continuity: {conv_layer.lipschitz_constant()}")
        # Number of input and output channels
        in_channels, out_channels, height, width = kernels.shape
        if num_kernels is None:
            num_kernels = in_channels + out_channels
        
        fig, axes = plt.subplots(min(num_kernels, out_channels), min(num_kernels, in_channels), figsize=(min(num_kernels, out_channels) * 2, min(num_kernels, in_channels) * 2))
        
        for i in range(min(num_kernels, out_channels)):
            for j in range(min(num_kernels, in_channels)):
                ax = axes[i, j] if out_channels > 1 else axes[j]
                kernel = kernels[j, i]
                ax.imshow(kernel, cmap='gray')
                ax.axis('off')
                ax.set_title(f"Kernel {i}-{j}")

        # Set the super title for the figure
        fig.suptitle(f'Layer {idx+1}', fontsize=4 * min(num_kernels, out_channels))

        # Save the figure
        save_path = os.path.join(save_dir, f'layer_{idx+1}_kernels.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to save memory

    print(f"Kernels saved to {save_dir}")

def visualize_network(model, input_tensor, num_kernels=6, save_dir='layer_visualizations'):
    # Ensure the model is in evaluation mode
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    # Save the original input image as layer_0
    original_image = input_tensor.squeeze().cpu().numpy()
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    plt.title('Original Image')
    plt.savefig(os.path.join(save_dir, 'layer_0_input.png'))
    plt.close()

    x = input_tensor.to(next(model.parameters()).device)
    
    with torch.no_grad():
        for idx, layer in tqdm(enumerate(model.network), total=len(model.network)):
            # Forward pass through the current layer
            x = layer(x)
            
            # Move the output to CPU and convert to numpy
            output = x.data.cpu().numpy()
            # Number of channels in the output
            channels = output.shape[1]
            
            if num_kernels is None:
                num_kernels = channels
            
            # Determine grid size for subplots
            num_cols = math.ceil(math.sqrt(min(num_kernels, channels)))
            num_rows = math.ceil(min(num_kernels, channels) / num_cols)

            # Plot the outputs
            if idx != len(model.network) - 1:
                fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
                axes = axes.flatten() if num_rows > 1 else [axes]
                
                for i in range(min(num_kernels, channels)):
                    ax = axes[i]
                    ax.imshow(output[0, i], cmap='gray')
                    ax.axis('off')
                    ax.set_title(f'Channel {i}')
                
                # Set the title for the figure
                fig.suptitle(f'Layer {idx+1}', fontsize=16)
                # Save the figure
                save_path = os.path.join(save_dir, f'layer_{idx+1}_output.png')
                plt.savefig(save_path)
                plt.close(fig)  # Close the figure to save memory
            else:
                plt.imshow(output[0, 0], cmap='gray')
                plt.axis('off')
                plt.title('Denoised Image')
                plt.savefig(os.path.join(save_dir, f'layer_{idx+1}_output.png'))
                plt.close()

    print(f"Layer outputs saved to {save_dir}")

def visualize_middle_output(model, input_tensor, save_dir='middle_visualizations', num_kernels=6, apply_first=False):
    # Ensure the model is in evaluation mode
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    x = input_tensor.to(next(model.parameters()).device)

    # Step through layers and get intermediate output
    with torch.no_grad():
        output = model.network[0](x)
        # output = x
        output = model.network[1].forward_half(output, apply_first=apply_first)
        
        # Move the output to CPU and convert to numpy
        output = output.data.cpu().numpy()
        channels = output.shape[1]  # Number of channels in the output

        # Number of kernels to visualize
        num_kernels = min(num_kernels, channels)

        # Determine grid size for subplots
        num_cols = math.ceil(math.sqrt(num_kernels))
        num_rows = math.ceil(num_kernels / num_cols)

        # Plot the outputs
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 6, num_rows * 6))
        axes = axes.flatten() if num_rows > 1 else [axes]

        for i in range(num_kernels):
            ax = axes[i]
            ax.imshow(output[0, i], cmap='gray')
            ax.axis('off')
            ax.set_title(f'Channel {i}')

        # Set the title for the figure
        fig.suptitle('Middle Layer Output', fontsize=8*num_cols)
        
        # Save the figure
        save_path = os.path.join(save_dir, 'middle_output.png')
        plt.savefig(save_path)
        plt.close(fig)  # Close the figure to save memory

    print(f"Middle layer output saved to {save_dir}")
    
    output = model(x).data.cpu().numpy()
    plt.imshow(output[0, 0], cmap='gray')
    plt.axis('off')
    plt.title('Denoised Image')
    plt.savefig(os.path.join(save_dir, f'output.png'))
    plt.close()
    print('Saved model output')
    
    output = model.network[0](x)
    output = model.network[1].forward_middle(output)
    output = model.network[2](output).data.cpu().numpy()
    plt.imshow(output[0, 0], cmap='gray')
    plt.axis('off')
    plt.title('Removed non linearities')
    plt.savefig(os.path.join(save_dir, f'output_no_nonlin.png'))
    plt.close()
    print('Saved model output without non linearities')


def visualize_learned_spline_activation(model, savedir='middle_visualizations', num_points=1000):
    """
    Visualize and save the learned spline activation function for each activation.

    Args:
    - model: The LinearSpline model instance.
    - filename: The file path where the plot will be saved.
    - num_points: Number of points to sample from the range [x_min, x_max].
    """
    # Generate a range of input values between x_min and x_max
    x_values = torch.linspace(model.x_min.item(), model.x_max.item(), num_points)

    # Get the spline activation function values for each activation
    activations = []
    for i in range(model.num_activations):
        coeffs = model.projected_coefficients[i].detach().cpu().numpy()
        spline_activation = np.interp(x_values.numpy(), 
                                      np.linspace(model.x_min.item(), model.x_max.item(), model.num_coeffs), 
                                      coeffs)
        activations.append(spline_activation)

    # Plot the activations
    num_activations = model.num_activations
    num_cols = int(np.sqrt(num_activations))  # Adjust this based on how many subplots you want in one row
    num_rows = (num_activations + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 3, num_rows * 3))
    axes = axes.flatten()

    for i in range(num_activations):
        ax = axes[i]
        ax.plot(x_values.numpy(), activations[i], color='black', linewidth=1)
        ax.set_title(f'Activation {i + 1}')
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(i + 1, len(axes)):  # Turn off extra subplots
        axes[j].axis('off')

    plt.suptitle('Learned Spline Activation Functions', fontsize=16)

    # Save the plot to a file
    plt.savefig(f'{savedir}/middle_activations.png', dpi=300)
    print(f"Plot saved to {f'{savedir}/middle_activations.png'}")
    
    # Show the plot
    plt.show()
    
# Assuming you have instantiated your model
exp_path = "/home/borna/Documents/parseval_conv/exps/sigma_25/symmirrorbcop_depth_16_width_16_kernel_3"

config = json.load(open(path.join(exp_path, 'config.json')))
# model = AveragedDenoiser(ParsevalCNN12(config['net_params'], config['activation_params']), config['net_params']['beta'])
model = ParsevalCNN22(config['net_params'], config['activation_params'])
model = model.to('cuda')

# Load the checkpoint
checkpoint = torch.load(path.join(exp_path, 'checkpoints/checkpoint.pth'))

# Load the state dictionary into the model
model.load_state_dict(checkpoint['state_dict'])
print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(config["exp_name"])

# Visualize and save the kernels
# visualize_kernels(model, save_dir=path.join(exp_path, 'kernel_visualizations'), num_kernels=None)

# Load and preprocess the image
image_path = 'bird-noisy.png'
image = Image.open(image_path).convert('L')  # Convert image to grayscale (L mode)

# Transform to tensor
transform = transforms.ToTensor()
input_tensor = transform(image).unsqueeze(0)  # Add a batch dimension

# Assuming `model` is already instantiated and loaded
# visualize_network(model.R if isinstance(model, AveragedDenoiser) else model, input_tensor, num_kernels=None, save_dir=path.join(exp_path, 'layer_visualizations'))

if isinstance(model, AveragedDenoiser):
    output = model(input_tensor.to(next(model.parameters()).device)).data.cpu().numpy()
    plt.imshow(output[0, 0], cmap='gray')
    plt.axis('off')
    plt.title('Denoised Image')
    plt.savefig(os.path.join(path.join(exp_path, 'layer_visualizations'), f'layer_{len(model.R.network)}_output.png'))
    plt.close()
    
    
# Get all SymSVD layers from the model
conv_layers = [layer for layer in model.network if isinstance(layer, LipschitzModuleL2)]
do_auto = False
# do_auto = True

if do_auto:
    num_kernels = None
    for idx, conv_layer in tqdm(enumerate(conv_layers), total=len(conv_layers), desc=f"Plotting kernel for each convoloution layer"):
        # Extract the weight of the convolutional kernel
        conv_layer._gen_weights()
        kernels = conv_layer.weight.data.cpu().numpy()
        conv_layer._input_shape = kernels.shape[:2]
        if True:
            print(f"Layer {idx} Lipschitz continuity: {conv_layer.lipschitz_constant()}")
        # Number of input and output channels
        in_channels, out_channels, height, width = kernels.shape
        if num_kernels is None:
            num_kernels = in_channels + out_channels
            
        print(torch.sigmoid(conv_layer.direction))
        print(torch.sigmoid(conv_layer.mask))
        
# visualize_middle_output(model, input_tensor, num_kernels=100, apply_first=False)

sz = 21
input_size = (1, 1, sz, sz)
delta_input = torch.zeros(input_size).to('cuda')
batch, channels, height, width = input_size
delta_input[:, :, height // 2, width // 2] = 1

visualize_middle_output(model, delta_input, num_kernels=100, apply_first=False)

middle = model.network[1]
tp = type(middle)

while(type(middle) == tp):
    middle = middle[1].middle_network
    
visualize_learned_spline_activation(middle)