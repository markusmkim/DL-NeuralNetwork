import numpy as np
import random
import math


class ImageGenerator:
    def __init__(self, size, centered, noise_rate):
        self.size = size
        self.centered = centered
        self.noise_rate = noise_rate

        # 0 = horizontal bars, 1 = vertical bars, 2 = cross, 3 = rectangle
        self.next_image_type = 0


    def generate_image_sets(self, number_of_images, share_train=0.7, share_validate=0.2):
        images = []
        for i in range(number_of_images):
            image = self.generate_image()
            images.append(image)
            self.next_image_type = (self.next_image_type + 1) % 4

        # split data into test, validate and test set
        return split_data_set(images, share_train, share_validate)


    def generate_image(self):
        if self.next_image_type == 0:
            return self.draw_horizontal_bars()

        if self.next_image_type == 1:
            return self.draw_vertical_bars()

        if self.next_image_type == 2:
            return self.draw_cross()

        return self.draw_rectangle()


    def draw_horizontal_bars(self):
        image = np.zeros((self.size, self.size))
        obj_center, x_radius, y_radius = self.generate_object_position(self.centered)
        for row in range(0, self.size, 2):
            for col in range(self.size):
                if abs(row - obj_center[1]) <= y_radius and \
                        abs(col - obj_center[0]) <= max(x_radius, self.size // 4):
                    image[row][col] = 1

                # add noise
                if random.random() < self.noise_rate:
                    # switch from 0 to 1 or 1 to 0
                    image[row][col] = (image[row][col] + 1) % 2

        return image


    def draw_vertical_bars(self):
        image = self.draw_horizontal_bars()
        return np.transpose(image)


    def draw_cross(self):
        image = np.zeros((self.size, self.size))
        obj_center, x_radius, y_radius = self.generate_object_position(self.centered)
        for row in range(self.size):
            for col in range(self.size):
                if (row == obj_center[1] and abs(obj_center[0] - col) <= x_radius) \
                        or (col == obj_center[0] and abs(obj_center[1] - row) <= y_radius):
                    image[row][col] = 1

                # add noise
                if random.random() < self.noise_rate:
                    # switch from 0 to 1 or 1 to 0
                    image[row][col] = (image[row][col] + 1) % 2

        return image


    def draw_rectangle(self):
        image = np.zeros((self.size, self.size))
        obj_center, x_radius, y_radius = self.generate_object_position(self.centered)
        for row in range(self.size):
            for col in range(self.size):
                if (abs(row - obj_center[1]) == y_radius and abs(col - obj_center[0]) <= x_radius) \
                        or (abs(col - obj_center[0]) == x_radius and abs(row - obj_center[1]) <= y_radius):
                    image[row][col] = 1

                # add noise
                if random.random() < self.noise_rate:
                    # switch from 0 to 1 or 1 to 0
                    image[row][col] = (image[row][col] + 1) % 2

        return image


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


def split_data_set(data, share_train, share_validate):
    train_size = math.floor(len(data)*share_train)
    val_size = len(data) - train_size if share_train + share_validate == 1 else math.floor(len(data)*share_validate)
    train = data[:train_size]
    val = data[train_size: train_size + val_size]
    test = data[train_size + val_size:]
    return train, val, test





