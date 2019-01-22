import cv2
from time import gmtime, strftime
import numpy as np
import math
import datetime
import time

def gimp_to_opencv_hsv(*hsv):
    """
    I use GIMP to visualize colors. This is a simple
    GIMP => CV2 HSV format converter.
    """
    return (hsv[0] / 2, hsv[1] / 100 * 255, hsv[2] / 100 * 255)

# colors
WHITE_LINES = { 'low_th': gimp_to_opencv_hsv(0, 0, 80),
                'high_th': gimp_to_opencv_hsv(359, 10, 100) }

BLUE_LINES = {'low_th': gimp_to_opencv_hsv(100, 150, 0),
               'high_th': gimp_to_opencv_hsv(140, 255, 255)}

COLORS = [BLUE_LINES]

class ImagePipeline:

    def __init__(self, image):

        self.image = image
        self.blank_image = np.zeros_like(image)

    def get_lane_lines_mask(self, hsv_image, colors):
        """
        Image binarization using a list of colors. The result is a binary mask
        which is a sum of binary masks for each color.
        """
        masks = []
        for color in colors:
            if 'low_th' in color and 'high_th' in color:
                mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
                if 'kernel' in color:
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
                masks.append(mask)
            else:
                raise Exception('High or low threshold values missing')
        if masks:
            return cv2.add(*masks)

    def get_binary_mask(self, image):
        # define range of blue color in HSV
        lower_blue = np.array([110, 50, 50])
        upper_blue = np.array([110, 255, 255])

        blue_mask = cv2.inRange(image, lower_blue, upper_blue)
        # define range of black color in HSV
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([170, 100, 50])
        # Threshold the HSV image to get only blue colors
        black_mask = cv2.inRange(image, lower_black, upper_black)

        # Bitwise-AND mask and original image
        # mask = cv2.bitwise_or(blue_mask, black_mask)
        mask = cv2.bitwise_or(blue_mask, black_mask)
        res = cv2.bitwise_and(image, image, mask=mask)
        return res

def get_lane_lines_mask(hsv_image, colors):
    """
    Image binarization using a list of colors. The result is a binary mask
    which is a sum of binary masks for each color.
    """
    masks = []
    for color in colors:
        if 'low_th' in color and 'high_th' in color:
            mask = cv2.inRange(hsv_image, color['low_th'], color['high_th'])
            if 'kernel' in color:
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, color['kernel'])
            masks.append(mask)
        else:
            raise Exception('High or low threshold values missing')
    if masks:
        return cv2.add(*masks)

# image is expected be in RGB color space
def select_rgb_blue_black(image):
    # define range of blue color in HSV
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([110, 255, 255])

    blue_mask = cv2.inRange(image, lower_blue, upper_blue)
    # define range of black color in HSV
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([170, 100, 50])
    # Threshold the HSV image to get only blue colors
    black_mask = cv2.inRange(image, lower_black, upper_black)

    # Bitwise-AND mask and original image
    # mask = cv2.bitwise_or(blue_mask, black_mask)
    mask = cv2.bitwise_or(blue_mask, black_mask)
    res = cv2.bitwise_and(image, image, mask= mask)
    return res

def convert_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

def convert_hls(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

def convert_gray_scale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_smoothing(image, kernel_size=15):
    """
    kernel_size must be postivie and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

def detect_edges(image, low_threshold=0, high_threshold=100):
    " Детектор границ Кенни "
    return cv2.Canny(image, low_threshold, high_threshold)


def filter_region(image, vertices):
    """
    Create the mask using the vertices and apply it to the input image
    """
    mask = np.zeros_like(image)
    if len(mask.shape) == 2:
        cv2.fillPoly(mask, vertices, 255)
    else:
        cv2.fillPoly(mask, vertices, (255,) * mask.shape[2])  # in case, the input image has a channel dimension
    return cv2.bitwise_and(image, mask)


def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return filter_region(image, vertices)


def hough_lines(image):
    """
    Detect straight lines

    `image` should be the output of a Canny transform.
    Returns hough lines (not the image with lines)
    """
    return cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=0, minLineLength=20, maxLineGap=30)

# Экстраполяция линий
def average_slope_intercept(lines):
    left_lines = []  # (slope, intercept)
    left_weights = []  # (length,)
    right_lines = []  # (slope, intercept)
    right_weights = []  # (length,)
    if lines.any():
        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue  # ignore a vertical line
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
                if slope < 0:  # y is reversed in image
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

    # add more weight to longer lines
    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

    return left_lane, right_lane  # (slope, intercept), (slope, intercept)


def make_line_points(y1, y2, line):
    """
    Convert a line represented in slope and intercept into pixel points
    """
    if line is None:
        return None

    slope, intercept = line

    # make sure everything is integer as cv2.line requires it
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)

    return ((x1, y1), (x2, y2))


def lane_lines(image, lines):
    left_lane, right_lane = average_slope_intercept(lines)

    y1 = image.shape[0]
    # y1 = image.get(3)
    y2 = y1 * 0.6  # slightly lower than the middle

    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line


def merge_lines(lines):
    "Объединяет несколько линий в одну"

    merged_line = lines
    if len(lines) > 1 and len(lines) < 3:
        left_line, right_line = lines
        if left_line is not None and right_line is not None:
                merged_line = ((
                    (
                    (left_line[0][0] + right_line[0][0]) // 2,
                    (left_line[0][1] + right_line[0][1]) // 2,
                    ),
                    (
                    (left_line[1][0] + right_line[1][0]) // 2,
                    (left_line[1][1] + right_line[1][1]) // 2,
                    )
                ))
        else:
            merged_line = left_line or right_line
    return (merged_line,)


def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line:
                l = [j for k in line if k for j in k]
                x1 = l[0]
                y1 = l[1]
                x2 = l[2]
                y2 = l[3]
                cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)


def draw_vertical_line(image, line, color=[255, 255, 255], thickness=20):
    " Рисует вертикальную линию для данной линии "
    line_image = np.zeros_like(image)
    if line is not None:
        l = [j for k in line for j in k]
        x1 = l[0]
        y1 = l[1]
        x2 = l[2]
        y2 = l[3]
        cv2.line(line_image, (x1, y1), (x1, y2), color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def get_angle(image, lines) -> float:
    alfa_angle = 0
    if lines is not None:
        for line in lines:
            if line:
                l = [j for k in line if k for j in k]
                x1 = l[0]
                y1 = l[1]
                x2 = l[2]
                y2 = l[3]
                # рисуем линию от центра изображения до центра
                WIDTH = 1280
                HEIGHT = 720

                # calc angles (converted in degrees)
                katet_a =  WIDTH // 2 - (x1 + x2) // 2
                katet_b = HEIGHT - (y1 + y2) // 2
                alfa_angle = math.atan(katet_a / katet_b) * (180 / 3.1415926)
                print(katet_a, katet_b, alfa_angle)

    return alfa_angle

def draw_binary_mask(binary_mask, img):
    if len(binary_mask.shape) != 2:
        raise Exception('binary_mask: not a 1-channel mask. Shape: {}'.format(str(binary_mask.shape)))
    masked_image = np.zeros_like(img)
    for i in range(3):
        masked_image[:,:,i] = binary_mask.copy()
    return masked_image

def image_pipeline(image):
    """
    Main image pipeline with 3 phases:
    * Raw image preprocessing and noise filtering;
    * Lane lines state update with the information gathered in preprocessing phase;
    * Drawing updated lane lines and other objects on image.
    """

    ### Phase 1: Image Preprocessing

    hsv_image = convert_hsv(image)
    binary_mask = get_lane_lines_mask(hsv_image, (BLUE_LINES, ))
    masked_image = draw_binary_mask(binary_mask, hsv_image)

    return masked_image
    # while True:
    #     ret, frame = cap.read()
    #     if ret:
    #         blue_black_filter = select_rgb_blue_black(frame)
    #         smooth_img = apply_smoothing(blue_black_filter, kernel_size=17)
    #         canny_img = detect_edges(smooth_img)
    #         roi_img = select_region(canny_img)
    #         found_lines = hough_lines(roi_img)
    #         lines = lane_lines(frame, found_lines)
    #         merged_line = merge_lines(lines)
    #         isCenterPosition(merged_line)
    #         drawed_lines = draw_lane_lines(frame, lines)
    #
    #         cv2.imshow('Video', drawed_lines)
    #         yield get_angle(frame, smooth_img)

if __name__ == "__main__":
    cap = cv2.VideoCapture(r'http://192.168.204.105:8081/?action=stream')

    if cap.isOpened():
        # get vcap property
        width = cap.get(3)  # float
        height = cap.get(4)  # float

    while True:
        ret, frame = cap.read()
        if ret:

            # blue_black_filter = get_lane_lines_mask(frame, COLORS)
            # smooth_img = apply_smoothing(blue_black_filter, kernel_size=17)
            # canny_img = detect_edges(smooth_img)
            # roi_img = select_region(canny_img)
            # found_lines = hough_lines(roi_img)
            # lines = lane_lines(frame, found_lines)
            # merged_line = merge_lines(lines)
            # isCenterPosition(merged_line)
            # drawed_lines = draw_lane_lines(frame, lines)
            drawed_lines = image_pipeline(frame)

            cv2.imshow('Video', drawed_lines)
            time.sleep(0.1)

        if cv2.waitKey(1) == 27:
            exit(0)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
    # img = cv2.imread(r'C:\Users\Artem\Pictures\vlcsnap-2019-01-21-21h10m52s718.png')
    # height, width, channels = img.shape
    #
    # if True:
    #     blue_black_filter = select_rgb_blue_black(img)
    #     smooth_img = apply_smoothing(blue_black_filter, kernel_size=17)
    #     canny_img = detect_edges(smooth_img)
    #     roi_img = select_region(canny_img)
    #     found_lines = hough_lines(roi_img)
    #     lines = lane_lines(img, found_lines)
    #     merged_line = merge_lines(lines)
    #     isCenterPosition(merged_line)
    #     drawed_lines = draw_lane_lines(img, merged_line)
    #
    #     cv2.imshow('Video', drawed_lines)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

