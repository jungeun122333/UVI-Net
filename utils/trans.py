class NumpyType(Base):
    def __init__(self, types, num=-1):
        self.types = types  # ('float32', 'int64')
        self.num = num

    def tf(self, img, k=0):
        if self.num > 0 and k >= self.num:
            return img
        # make this work with both Tensor and Numpy
        return img.astype(self.types[k])

    def __str__(self):
        s = ", ".join([str(s) for s in self.types])
        return "NumpyType(({}))".format(s)