from LtVisualizer.tensor import Tensor

x = Tensor(1, 2, 3, 4)
print(x)

x.reflect_on_x()
print(x)

x.reflect_on_y()
print(x)
