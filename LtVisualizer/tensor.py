import numpy as np
import matplotlib.pyplot as plt


class Tensor:
    def __init__(self, x1: int, y1: int, x2: int, y2: int):
        if not all(isinstance(arg, int) for arg in (x1, x2, y1, y2)):
            raise TypeError(
                f"All input must be integers. Input received {type(x1).__name__}, {type(y1).__name__}, {type(x2).__name__}, {type(y2).__name__}"
            )
        self.obj = np.array([[x1, y1], [x2, y2]])

    def __repr__(self):
        return (
            f"[{self.obj[0, 0]} {self.obj[0, 1]} \n {self.obj[1, 0]} {self.obj[1, 1]}]"
        )

    def reflect_on_x(self):
        self.obj[0, 1] *= -1
        self.obj[1, 1] *= -1
        return self

    def reflect_on_y(self):
        self.obj[0, 0] *= -1
        self.obj[1, 0] *= -1
        return self

    def scaling_all_direction(self, m: int):
        # scaling the graph my m
        self.obj[0, 0] *= m
        self.obj[0, 1] *= m
        self.obj[1, 0] *= m
        self.obj[1, 1] *= m
        return self

    def horizontal_sheer(self, m: int):
        self.obj[0, 0] += m * self.obj[0, 1]
        self.obj[1, 0] += m * self.obj[1, 1]
        return self

    def vertical_sheer(self, m: int):
        self.obj[0, 1] += m * self.obj[0, 0]
        self.obj[1, 1] += m * self.obj[1, 0]
        return self

    def copy(self):
        new_t = Tensor(0, 0, 0, 0)
        new_t.obj = self.obj.copy()
        return new_t


def visualize(tensors, titles=None):
    """
    tensors: List of Tensor objects
    titles: List of strings for plot titles
    """
    n = len(tensors)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    # Calculate limits for consistent view
    all_coords = np.concatenate([t.obj.flatten() for t in tensors])
    limit = max(abs(all_coords.min()), abs(all_coords.max())) + 1

    for i, t in enumerate(tensors):
        ax = axes[i]
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Plot the two vectors
        colors = ["red", "blue"]
        for j in range(2):
            ax.quiver(
                0,
                0,
                t.obj[j, 0],
                t.obj[j, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=colors[j],
                label=f"Vector {j + 1}",
            )

        ax.set_title(titles[i] if titles else f"State {i}")
        ax.legend()

    plt.tight_layout()
    plt.show()
