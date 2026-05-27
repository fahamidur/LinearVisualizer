import numpy as np


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
