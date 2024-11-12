import numpy as np

def depthwise_convolution(image, depthwise_filter):
    """
    Perform depthwise convolution on an image with separate filters for each channel.
    """
    channels, height, width = image.shape
    filter_height, filter_width = depthwise_filter.shape[1:]
    
    output_height = height - filter_height + 1
    output_width = width - filter_width + 1
    depthwise_output = np.zeros((channels, output_height, output_width))
    
    # Apply the filter to each channel independently
    for c in range(channels):
        for i in range(output_height):
            for j in range(output_width):
                region = image[c, i:i+filter_height, j:j+filter_width]
                depthwise_output[c, i, j] = np.sum(region * depthwise_filter[c])
    
    return depthwise_output

def pointwise_convolution(depthwise_output, pointwise_filter):
    """
    Perform pointwise convolution on the output of depthwise convolution.
    """
    channels, height, width = depthwise_output.shape
    num_filters = pointwise_filter.shape[0]
    pointwise_output = np.zeros((num_filters, height, width))
    
    # Apply the 1x1 filter across all channels
    for n in range(num_filters):
        for c in range(channels):
            pointwise_output[n] += depthwise_output[c] * pointwise_filter[n, c]
    
    return pointwise_output

def convolution(image, kernel, mode='depthwise'):
    """
    Function to decide between depthwise and pointwise convolution based on the flag.
    
    Parameters:
    - image: Input image (3D array with channels, height, width).
    - kernel: Filter to apply (either for depthwise or pointwise).
    - mode: 'depthwise' or 'pointwise' (default is 'depthwise').
    """
    if mode == 'depthwise':
        # Ensure kernel shape matches the number of channels in the image
        if kernel.shape[0] != image.shape[0]:
            raise ValueError("Depthwise filter must have the same number of channels as the input image.")
        return depthwise_convolution(image, kernel)
    elif mode == 'pointwise':
        # Apply pointwise convolution
        return pointwise_convolution(image, kernel)
    else:
        raise ValueError("Invalid mode. Choose either 'depthwise' or 'pointwise'.")

# Example Usage:
image = np.random.rand(3, 6, 6)  # Input image with 3 channels, 6x6 size

# Depthwise convolution filter: 3 filters (one for each channel), each of size 3x3
depthwise_filter = np.random.rand(3, 3, 3)

# Pointwise convolution filter: 2 filters (output depth of 2), operating on 3 input channels
pointwise_filter = np.random.rand(2, 3)

# Choose the mode ('depthwise' or 'pointwise')
mode = input("Enter the mode (depthwise/pointwise): ").strip().lower()

# Perform the convolution based on the mode
if mode == 'depthwise':
    output = convolution(image, depthwise_filter, mode='depthwise')
    print("Depthwise Convolution Output:\n", output)
elif mode == 'pointwise':
    # First, perform depthwise convolution to get the intermediate output
    depthwise_output = depthwise_convolution(image, depthwise_filter)
    output = convolution(depthwise_output, pointwise_filter, mode='pointwise')
    print("Pointwise Convolution Output:\n", output)
else:
    print("Invalid mode selected.")
