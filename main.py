from LtVisualizer.tensor import Tensor, visualize

# 1. Start with an original
original = Tensor(1, 2, 3, 4)

# 2. Build the chain of copies
history = [
    original.copy(),
    original.copy().reflect_on_x(),
    original.copy().reflect_on_x().reflect_on_y(),
    original.copy().reflect_on_x().reflect_on_y().scaling_all_direction(2),
]

# 3. Visualize
visualize(history, titles=["Original", "Reflect X", "Reflect XY", "Scaled"])
