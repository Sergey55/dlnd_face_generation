class Scale(object):
    """ Takes an image in range 0-1 and rescales it to desired range"""

    def __init__(self, range=(-1, 1)):
        self.min, self.max = range

    def __call__(self, image):
        image = image * (self.max - self.min) + self.min

        return image

