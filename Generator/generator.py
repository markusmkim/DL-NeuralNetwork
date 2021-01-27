import numpy as np


def samplemat(dims):
    """Make a matrix with all zeros and increasing elements on the diagonal"""
    aa = np.zeros(dims)
    for i in range(min(dims)):
        aa[i, i] = 2
    return aa


class ImageGenerator:
    def __init__(self, size, centered=True):
        self.size = size
        self.centered = centered

        # 0 = horizontal bars, 1 = vertical bars, 2 = cross, 3 = rectangle
        self.next_image_type = 0


    def generate_image_sets(self, number_of_images, training_set=0.7, validate_set=0.2, test_set=0.1):
        if not self.centered:
            return

        images = []
        for i in range(number_of_images):
            image = self.generate_image()
            images.append(image)
            self.next_image_type = (self.next_image_type + 1) % 4

        # split into training, validate and test sets

        return images


    def generate_image(self):

        if self.next_image_type == 0:
            return self.draw_horizontal_bars()

        if self.next_image_type == 0:
            return self.draw_vertical_bars()

        if self.next_image_type == 0:
            return self.draw_cross()

        return self.draw_rectangle()


    def draw_horizontal_bars(self):
        image = np.zeros((self.size, self.size))
        for i in range(0, self.size, 2):
            image[i] = np.ones(self.size)
        return image


    def draw_vertical_bars(self):
        image = self.draw_horizontal_bars()
        return np.transpose(image)


    def draw_cross(self):
        image = np.zeros((self.size, self.size))
        middle_point = self.size / 2
        cross_pixel_index = middle_point
        cross_arm_len = max(middle_point - 2, 1)
        for row in range(self.size):
            for column in range(self.size):
                if (row == cross_pixel_index and abs(middle_point - column) <= cross_arm_len) \
                        or (column == self.size - cross_pixel_index and abs(middle_point - row) <= cross_arm_len):
                    image[row][column] = 1

        return image


    def draw_rectangle(self):
        image = np.zeros((self.size, self.size))
        return 0



