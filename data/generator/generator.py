import numpy as np
import random
from data.generator.utils import split_data_set, one_hot_encoder


class ImageGenerator:
    def __init__(self, size, two_dim=True, centered=False, noise_rate=0.1):
        self.size = size
        self.two_dimensions = two_dim
        self.centered = centered
        self.noise_rate = noise_rate

        # 0 = horizontal bars, 1 = vertical bars, 2 = cross, 3 = rectangle (if two dimensions)
        self.next_image_type = 0


    def generate_image_sets(self, number_of_images, share_train=0.7, share_validate=0.1, flatten=False):
        images = []
        for i in range(number_of_images):
            image = self.generate_2d_image(flatten) if self.two_dimensions else self.generate_1d_image()
            target = one_hot_encoder(self.next_image_type, 4)
            images.append((image, target))
            self.next_image_type = (self.next_image_type + 1) % 4

        # split data into test, validate and test set
        return split_data_set(images, share_train, share_validate)


    def generate_1d_image(self):
        image = np.zeros(self.size)
        image_type = self.next_image_type + 1
        pos_bit_prob = 2 * (image_type + 1) / (self.size - 2)
        pos_bits_indexes = []
        for i in range(0, self.size, 2):
            if random.random() < pos_bit_prob:
                pos_bits_indexes.append(i)
            if len(pos_bits_indexes) == image_type:
                break
        if len(pos_bits_indexes) < image_type:
            return self.generate_1d_image()

        for i in range(len(pos_bits_indexes)):
            if i == len(pos_bits_indexes) - 1:
                possible_pos_seg_len = max(self.size - pos_bits_indexes[i] - 2, 0)
            else:
                possible_pos_seg_len = max(pos_bits_indexes[i + 1] - pos_bits_indexes[i] - 2, 0)
            pos_seg_len = random.randint(0, possible_pos_seg_len)

            for j in range(pos_seg_len + 1):
                image_index = pos_bits_indexes[i]
                image[image_index + j] = 1
        return image


    def generate_2d_image(self, flatten):
        if self.next_image_type == 0:
            return self.draw_horizontal_bars(flatten)

        if self.next_image_type == 1:
            return self.draw_vertical_bars(flatten)

        if self.next_image_type == 2:
            return self.draw_cross(flatten)

        return self.draw_rectangle(flatten)


    def draw_horizontal_bars(self, flatten):
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

        return np.ravel(image) if flatten else image


    def draw_vertical_bars(self, flatten):
        image = self.draw_horizontal_bars(False)
        return np.ravel(np.transpose(image)) if flatten else np.transpose(image)


    def draw_cross(self, flatten):
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

        return np.ravel(image) if flatten else image


    def draw_rectangle(self, flatten):
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

        return np.ravel(image) if flatten else image


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
