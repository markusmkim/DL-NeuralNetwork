import numpy as np
import random

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
        print('Generating cross')
        image = np.zeros((self.size, self.size))
        obj_center, x_radius, y_radius = self.generate_object_position(self.centered)
        print(obj_center, x_radius, y_radius)
        for row in range(self.size):
            for column in range(self.size):
                if (row == obj_center[1] and abs(obj_center[0] - column) <= x_radius) \
                        or (column == obj_center[0] and abs(obj_center[1] - row) <= y_radius):
                    image[row][column] = 1

        return image


    def draw_rectangle(self):
        image = np.zeros((self.size, self.size))
        return 0


    def generate_object_position(self, centered):
        image_center = self.size // 2

        # either center objects
        if centered:
            obj_center_x_co, obj_center_y_co = image_center, image_center

        # or generate random position
        else:
            obj_center_x_co = image_center + random.randint(- (image_center // 2), image_center // 2)
            obj_center_y_co = image_center + random.randint(- (image_center // 2), image_center // 2)

        x_radius = random.randint(image_center // 4, image_center - abs(image_center - obj_center_x_co) - 1)
        y_radius = random.randint(image_center // 4, image_center - abs(image_center - obj_center_y_co) - 1)

        return (obj_center_x_co, obj_center_y_co), x_radius, y_radius



