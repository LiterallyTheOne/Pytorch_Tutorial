import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter

# ---------------------------
# Parameters
# ---------------------------
image_size = (6, 6)
kernel_size = (3, 3)
stride = 1
padding = 1
dilation = 1  # try 2 to see dilation effect

# Create dummy input
image = np.arange(image_size[0] * image_size[1]).reshape(image_size)

# Effective kernel size after dilation
dilated_kernel_size = (
    kernel_size[0] + (kernel_size[0] - 1) * (dilation - 1),
    kernel_size[1] + (kernel_size[1] - 1) * (dilation - 1),
)

# Pad the image
padded_image = np.pad(image, pad_width=padding, mode="constant", constant_values=0)

# Compute number of steps (output feature map size)
out_h = (padded_image.shape[0] - dilated_kernel_size[0]) // stride + 1
out_w = (padded_image.shape[1] - dilated_kernel_size[1]) // stride + 1

# ---------------------------
# Plot setup
# ---------------------------
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xticks(np.arange(-0.5, padded_image.shape[1], 1))
ax.set_yticks(np.arange(-0.5, padded_image.shape[0], 1))
ax.grid(True)
ax.imshow(padded_image, cmap="Blues")

# Add text labels
for i in range(padded_image.shape[0]):
    for j in range(padded_image.shape[1]):
        ax.text(j, i, str(padded_image[i, j]), ha="center", va="center")

# Rectangle for kernel
rect = patches.Rectangle((0, 0), dilated_kernel_size[1] - 1e-6, dilated_kernel_size[0] - 1e-6,
                         linewidth=2, edgecolor="red", facecolor="none")
ax.add_patch(rect)


# ---------------------------
# Animation function
# ---------------------------
def update(frame):
    i = frame // out_w
    j = frame % out_w
    y = i * stride
    x = j * stride
    rect.set_xy((x - 0.5, y - 0.5))  # shift rectangle
    return rect,


# ---------------------------
# Create animation
# ---------------------------
frames = out_h * out_w
ani = FuncAnimation(fig, update, frames=frames, blit=True, repeat=False)

# Save to GIF
ani.save("convolution.gif", writer=PillowWriter(fps=2))

plt.show()
