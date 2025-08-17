import matplotlib.pyplot as plt


def draw_mlp(layers: list[int]):
    """
    Draw a simple fully connected neural network diagram.
    layers: list of neuron counts per layer (e.g. [8, 16, 4])
    """
    fig, ax = plt.subplots()
    ax.axis('off')

    v_spacing = 1.0  # vertical spacing between neurons
    h_spacing = 2.0  # horizontal spacing between layers

    # positions of neurons
    positions = {}
    for i, layer_size in enumerate(layers):
        layer_top = v_spacing * (layer_size - 1) / 2
        for j in range(layer_size):
            positions[(i, j)] = (i * h_spacing, layer_top - j * v_spacing)

    # draw edges
    for i, layer_size in enumerate(layers[:-1]):
        for j in range(layer_size):
            for k in range(layers[i + 1]):
                x1, y1 = positions[(i, j)]
                x2, y2 = positions[(i + 1, k)]
                ax.plot([x1, x2], [y1, y2], "-", linewidth=0.5)

    # draw neurons
    for (i, j), (x, y) in positions.items():
        circle = plt.Circle((x, y), radius=0.1, color="w", ec="k", zorder=3)
        ax.add_patch(circle)

    plt.gca().set_facecolor("black")
    plt.show()

    file_name = "model-"
    file_name += "-".join(map(str, layers))
    file_name += ".webp"

    fig.savefig(file_name)


def main():
    layers = [8, 4]
    draw_mlp(layers)


if __name__ == '__main__':
    main()
