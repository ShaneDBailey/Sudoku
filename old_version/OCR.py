"""
Author: Shane Bailey

Library Resources:
- pip install numpy
- pip install opencv-python
- pip install easyocr

Citations:
- none

Important References:
- https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

"""
#external libraries
import numpy
import cv2
#python libraries
import os
import tensorflow as tf

loaded_model = tf.keras.models.load_model('hard_model.keras')
#-------------------------Cell_class------------------------------------
class Cell:
    def __init__(self, image,stat,id):
        self.image = image
        self.id = id
        self.text = None
        self.topleft = (stat[0],stat[1])
        self.stats = stat
        self.predict_text()
        #print(self.text)
    def predict_text(self):
        # Preprocess the image
        processed_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        processed_image = cv2.resize(processed_image, (28, 28))
        processed_image = processed_image.reshape(1, 28, 28) / 255.0
        
        # Predict using the model
        prediction = loaded_model.predict(processed_image)
        predicted_class = numpy.argmax(prediction)
        
        self.text = str(predicted_class)
    def __repr__(self):
        return self.text

def detect_sudoku_board(image):
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
        
        #print("Bounding Rectangle: x={}, y={}, w={}, h={}".format(x, y, w, h))
        file_path = os.path.join("debugging", "wholepuzzle.png")
        cv2.imwrite(file_path, image[y:y+h, x:x+w])
        return image[y:y+h, x:x+w], x, y
#------------------------Cell_finder-------------------------------------
def detect_boxes(image,line_min_width = 50):
    #---converts image to greyscale
    gray_scale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("gray_scale",gray_scale)
    #---goes through the image, either makes it black or white, gets rid of the grey
    #---variable declaration so I can use this function like this
    _,black_white_bin=cv2.threshold(gray_scale,230,255,cv2.THRESH_BINARY)
    #cv2.imshow("black_white_threshold",black_white_bin)
    #---These mark the shape of dectection they are more open up to than just lines
    #---defines an area of a line 1 x line_min_width in terms of pixels for horizontal
    kernal_h=numpy.ones((1,line_min_width), numpy.uint8)
    #---defines an area of a line line_min_width x 1 in terms of pixels for vertical
    kernal_v=numpy.ones((line_min_width,1), numpy.uint8)
    #---grabs all the kernals, removes noise and compiles it into an image
    img_bin_h=cv2.morphologyEx(~black_white_bin, cv2.MORPH_OPEN, kernal_h)
    img_bin_v=cv2.morphologyEx(~black_white_bin, cv2.MORPH_OPEN, kernal_v)
    #---grabs both images of bin_h and bin_v and compiles it to one
    img_bin_final=img_bin_h|img_bin_v
    #This thickens the lines
    final_kernel=numpy.ones((3,3), numpy.uint8)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=2)
    img_bin_final=cv2.dilate(img_bin_final,final_kernel,iterations=2)
    #cv2.imshow("kernel_results",~img_bin_final)
    ret, labels, stats,centroids = cv2.connectedComponentsWithStats(~img_bin_final, connectivity=4, ltype=cv2.CV_32S)
    return stats
#------------------------Spliced_cells_into_set_of_smaller_iamges--------
def splice_image(image, stats):
    spliced_images = []
    image_height = image.shape[0]
    id = 0

    for i, stat in enumerate(stats):
        x, y, w, h = stat[0:4]

        if h <= 0.20 * image_height and w <= 0.20 * image_height and h >= 0.08 * image_height and w >= 0.08 * image_height:
            cropped_image = image[y:y+h, x:x+w]
            spliced_images.append(Cell(cropped_image, stat, id))
            id += 1

    return spliced_images

def save_spliced_images(spliced_images,folder_path):
    for i, spliced_image in enumerate(spliced_images):
        file_path = os.path.join(folder_path, f"image_{i}.png")
        cv2.imwrite(file_path, spliced_image.image)


def process_images(input_folder, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate through each image file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)
            board_image, x, y = detect_sudoku_board(image)
            stats = detect_boxes(board_image)
            cells = splice_image(board_image,stats)

            for cell in cells:
                gray_image = cv2.cvtColor(cell.image, cv2.COLOR_BGR2GRAY)
                _,black_white_bin=cv2.threshold(gray_image,200,255,cv2.THRESH_BINARY)
                cv2.imshow("black_white_threshold",black_white_bin)
                white_threshold = 230
                black_pixels = numpy.sum(white_threshold < black_white_bin)
                total_pixels = cell.image.size
                #print(total_pixels)
                black_percentage = black_pixels / total_pixels
                #print(black_pixels)
                output_path = os.path.join(output_folder, filename + "_" + str(cell.id) + ".png")
                cv2.imwrite(output_path, black_white_bin)


def return_puzzle(image):
    board_image, x_offset, y_offset = detect_sudoku_board(image)
    if board_image is not None and len(board_image) > 0:
        stats = detect_boxes(board_image)
        cells = splice_image(board_image,stats)

        return cells, x_offset, y_offset
    return None

cv2.waitKey(0)
cv2.destroyAllWindows()