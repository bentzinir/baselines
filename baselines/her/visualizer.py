import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np


class VisObserver:
    def __init__(self, explorer=None, sharing=None, exploration=None, goaler=None):
        self.eta = 0.01
        plt.figure(f"explorer: {explorer}, "
                   f"sharing: {sharing}, "
                   f"exploration: {exploration}, "
                   f"goaler: {goaler}")
        # self.im = plt.imshow(state_image)
        self.scat = plt.scatter(x=[], y=[], c='r', marker='+')
        plt.show(block=False)
        plt.tight_layout()
        self.state_image = None
        self.im = None

    def update(self, image_t, points):
        if self.state_image is None:
            self.state_image = image_t
            self.im = plt.imshow(self.state_image, norm=colors.LogNorm(vmin=0.01, vmax=self.state_image.max()))
        else:
            self.state_image = (1 - self.eta) * self.state_image + self.eta * image_t
            # self.state_image += image_t
            # self.state_image[self.state_image > 0] = 1
        self.im.set_data(self.state_image)
        self.im.set_clim(vmin=self.state_image.min() + 0.01, vmax=self.state_image.max())

        if points:
            self.scat.set_offsets(self.points2xy(points))
        plt.pause(0.0001)
        plt.draw()

    def points2xy(self, x):
        if len(x[0]) == 2:
            return 84 * np.flip(x, axis=1)
        a = 84 * np.asarray(x)[:, 4:6]
        return np.flip(a, axis=1)