"""
Author: Shane Bailey

Library Resources:
- pip install numpy
- pip install opencv-python
- pip install tensorflow

Important References:
- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""
#external libraries
import numpy
import cv2
import tensorflow as tf
BOARD_LENGTH = 9
loaded_model = tf.keras.models.load_model('hard_model.keras')
#-------------------------Cell_class------------------------------------
class Cell:
    def __init__(self, image,stat,id, x_offset, y_offset):
        self.image = image
        self.id = id
        self.text = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.stats = stat
        self.predict_text()

    def predict_text(self):
        # Preprocess the image
        processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.resize(processed_image, (28, 28))
        processed_image = processed_image.reshape(1, 28, 28) / 255.0
        
        # Predict using the model
        prediction = loaded_model.predict(processed_image)
        predicted_class = numpy.argmax(prediction)
        
        self.text = str(predicted_class)

class sudoku_puzzle:
    def __init__(self, image):
        self.board_image = None
        self.board_x_offset = None
        self.board_y_offset = None
        self.cell_list = None
        self.detect_sudoku_board(image)
        self.return_puzzle()

    def detect_sudoku_board(self,image):
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, black_white_image = cv2.threshold(gray_scale, 254, 255, cv2.THRESH_BINARY)

        contours, hierachy = cv2.findContours(black_white_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_square_contour = None
        max_area = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 0.9 < aspect_ratio < 1.1:
                area = cv2.contourArea(contour)

                if area > max_area:
                    max_area = area
                    biggest_square_contour = contour

        if biggest_square_contour is not None:
            x, y, w, h = cv2.boundingRect(biggest_square_contour)

            self.board_image = image[y:y+h, x:x+w]
            self.board_x_offset = x
            self.board_y_offset = y

    def splice_image(self, image, stats):
        spliced_images = []
        image_height = image.shape[0]
        id = 0

        for stat in stats:
            x, y, w, h = stat[0:4]

            if h <= 0.20 * image_height and w <= 0.20 * image_height and h >= 0.07 * image_height and w >= 0.07 * image_height:
                cropped_image = image[y:y+h, x:x+w]
                spliced_images.append(Cell(cropped_image, stat, id, self.board_x_offset + x, self.board_y_offset + y))
                id += 1

        self.cell_list = spliced_images

    def return_puzzle(self):
        if self.board_image is not None and len(self.board_image) > 0:
            stats = detect_boxes(self.board_image)
            self.splice_image(self.board_image,stats)
#------------------------Cell_finder-------------------------------------
def detect_boxes(image,line_min_width = 50):
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    _,black_white_bin=cv2.threshold(gray_scale,230,255,cv2.THRESH_BINARY)
    kernal_h=numpy.ones((1,line_min_width), numpy.uint8)
    kernal_v=numpy.ones((line_min_width,1), numpy.uint8)
    img_bin_h=cv2.morphologyEx(~black_white_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v=cv2.morphologyEx(~black_white_bin, cv2.MORPH_OPEN, kernal_v)
    img_bin_final=img_bin_h|img_bin_v
    final_kernel=numpy.ones((3,3), numpy.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=2)
    _, _, stats, _ = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=4, ltype=cv2.CV_32S)
    return stats

