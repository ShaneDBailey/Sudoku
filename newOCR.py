"""
Author: Shane Bailey

File Description:
- This file serves to do image recongnition and return a sudoku board
- It returns the board as a list of cells, order from left to right: top to bottom

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
import tensorflow
#load in the number recongition model
number_recongition_model = tensorflow.keras.models.load_model('hard_model.keras')
#-------------------------Cell_class------------------------------------
# The cell class is to represent a square within side the sudoku board
# We need to know what number is there and its offset,
#           So we can later input them when its solved
class Cell:
    def __init__(self, image, stat, x_offset, y_offset):
        self.cell_image = image
        self.text = None
        self.cell_x_offset = x_offset
        self.cell_y_offset = y_offset
        self.bounding_box_information = stat
        self.predict_text()

    def predict_text(self):
        # Gray scales the cell image and resizes it to be 28 by 28 for the image recongnition
        processed_image = cv2.cvtColor(self.cell_image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.resize(processed_image, (28, 28))
        processed_image = processed_image.reshape(1, 28, 28) / 255
        
        # Recongnize the number in the image
        number_recongized = number_recongition_model.predict(processed_image)
        self.text = numpy.argmax(number_recongized)

#-------------------------Sudoku_class------------------------------------
# The sudoku class is to represent the sudoku board
# It contains its image, its offset from the screen_shot
# and a list of cells 
class sudoku_puzzle:
    def __init__(self, image):
        self.board_image = None
        self.board_x_offset = None
        self.board_y_offset = None
        self.cell_list = None
        self.detect_sudoku_board(image)

    def detect_sudoku_board(self,image):
        #gray scales the image and then does a black and white threshold
        gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, black_white_image = cv2.threshold(gray_scale, 254, 255, cv2.THRESH_BINARY)
        #finds contours, contours being connected lines, in this case squares ie the cells
        contours, _ = cv2.findContours(black_white_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        biggest_square_contour = None
        max_area = 0
        #find the biggest square contour (the sudoku board)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)

            if 0.9 < aspect_ratio < 1.1:
                area = cv2.contourArea(contour)

                if area > max_area:
                    max_area = area
                    biggest_square_contour = contour
        #if we do find atleast one square contour gather all the board info
        if biggest_square_contour is not None:
            x, y, w, h = cv2.boundingRect(biggest_square_contour)

            self.board_image = image[y:y+h, x:x+w]
            self.board_x_offset = x
            self.board_y_offset = y
            stats = detect_boxes(self.board_image)
            self.splice_image(stats)

    def splice_image(self,stats):
        spliced_images = []
        image_height = self.board_image.shape[0]

        for stat in stats:
            x, y, w, h = stat[0:4]

            if h <= 0.20 * image_height:
                cropped_image = self.board_image[y:y+h, x:x+w]
                spliced_images.append(Cell(cropped_image, stat, self.board_x_offset + x, self.board_y_offset + y))

        self.cell_list = spliced_images
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