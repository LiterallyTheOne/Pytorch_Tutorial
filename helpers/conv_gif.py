import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------
# Parameters
# ---------------------------
image_size = (6, 6)
kernel = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])  # simple edge-detection kernel

stride = 1
padding = 1
dilation = 1  # try 2 for dilation effect

# Create dummy input
image = np.arange(image_size[0] * image_size[1]).reshape(image_size)

# Effective kernel size after dilation
dilated_kernel_size = (
    kernel.shape[0] + (kernel.shape[0] - 1) * (dilation - 1),
    kernel.shape[1] + (kernel.shape[1] - 1) * (dilation - 1),
)

# Pad the image
padded_image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)

# Output dimensions
out_h = (padded_image.shape[0] - dilated_kernel_size[0]) // stride + 1
out_w = (padded_image.shape[1] - dilated_kernel_size[1]) // stride + 1
output = np.zeros((out_h, out_w))

# ---------------------------
# Figure setup
# ---------------------------
fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(10, 5))

# Input image with grid
ax_in.set_title("Input (with padding)")
ax_in.set_xticks(np.arange(-0.5, padded_image.shape[1], 1))
ax_in.set_yticks(np.arange(-0.5, padded_image.shape[0], 1))
ax_in.grid(True)
ax_in.imshow(padded_image, cmap="Blues")

# Write input numbers
for i in range(padded_image.shape[0]):
    for j in range(padded_image.shape[1]):
        ax_in.text(j, i, str(padded_image[i, j]), ha="center", va="center")

# Rectangle showing kernel
rect = patches.Rectangle((0, 0),
                         dilated_kernel_size[1] - 1e-6,
                         dilated_kernel_size[0] - 1e-6,
                         linewidth=2, edgecolor="red", facecolor="none")
ax_in.add_patch(rect)

# Output feature map
ax_out.set_title("Output (feature map)")
ax_out.set_xticks(np.arange(-0.5, out_w, 1))
ax_out.set_yticks(np.arange(-0.5, out_h, 1))
ax_out.grid(True)
im_out = ax_out.imshow(output, cmap="Greens", vmin=-10, vmax=10)

# Text annotations for output
texts_out = [[ax_out.text(j, i, "", ha="center", va="center") for j in range(out_w)] for i in range(out_h)]

# Rectangle to highlight output cell
rect_out = patches.Rectangle((0 - 0.5, 0 - 0.5), 1, 1,
                             linewidth=2, edgecolor="red", facecolor="none")
ax_out.add_patch(rect_out)


# ---------------------------
# Convolution calculation
# ---------------------------
def conv_at(i, j):
    """Compute convolution value at output[i,j]."""
    y = i * stride
    x = j * stride
    region = padded_image[y:y + dilated_kernel_size[0]:dilation,
    x:x + dilated_kernel_size[1]:dilation]
    return np.sum(region * kernel)


# ---------------------------
# Animation update
# ---------------------------
def update(frame):
    i = frame // out_w
    j = frame % out_w

    # Move kernel rect
    y = i * stride
    x = j * stride
    rect.set_xy((x - 0.5, y - 0.5))

    # Compute convolution result
    val = conv_at(i, j)
    output[i, j] = val

    # Update output heatmap
    im_out.set_data(output)

    # Update output text
    for ii in range(out_h):
        for jj in range(out_w):
            if output[ii, jj] != 0:
                texts_out[ii][jj].set_text(str(int(output[ii, jj])))
            else:
                texts_out[ii][jj].set_text("")

    # Move output highlight
    rect_out.set_xy((j - 0.5, i - 0.5))

    return rect, im_out, rect_out


# ---------------------------
# Run animation
# ---------------------------
frames = out_h * out_w
ani = FuncAnimation(fig, update, frames=frames, blit=False, repeat=False)
ani.save("convolution_with_output.gif", writer=PillowWriter(fps=2))

plt.show()
