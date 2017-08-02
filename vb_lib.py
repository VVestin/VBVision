import cv2
import copy
import logging
import math
import numpy as np
import os
import pickle
import skvideo.io

# Color constants in hsv colorspace
HSV_COLOR = {
    'RED': (0, 255, 255),
    'ORANGE': (10, 255, 255),
    'YELLOW': (25, 255, 255),
    'GREEN': (65, 255, 255),
    'CYAN': (90, 255, 255),
    'BLUE': (105, 255, 255),
    'PURPLE': (140, 255, 255),
    'PINK': (150, 255, 255),
    'WHITE': (0, 0, 255),
    'BLACK': (0, 255, 0)
}

# Image analysis constants:
MIN_CONTOUR_AREA = 300 # Minimum area for a contour to be considered trackable
MIN_BALL_LIFETIME = 15 # Minimum number of frames a contour can appear on to be a ball candidate
MIN_FOUND_RATIO = .8 # Minimum ratio of frames a contour is found in to its lifetime
MIN_POLY_COEFF = -2 # Minimum value the highest degree coefficient can take in the polynomial fit
MAX_POLY_COEFF = 2 # Maximum value the highest degree coefficient can take in the polynomial fit
MAX_UNFOUND_STRIKES = 12 # Maximum number of frames a contour can be missing in while still tracked

# Ball classifier features:
# radius, color, roundness, poly residual, height, arc_length, 
# avg 2nd difference, avg direction change, contour area, lifetime
BALL_WEIGHTS = [
        3,  # radius
        3,  # color
        10, # roundness
        15, # poly residual
        25, # height
        15, # arc_length
        15, # avg 1st difference
        5,  # avg 2nd difference
        15, # avg direction change
        3,  # contour area
        13] # lifetime
# Bounds on the features, 1st element is weighted as 0, 2nd element weighted as 1, with linear interpolation inbetween
BALL_FEATURE_BOUNDS = [
        (5, .1),    # radius
        (0, .7),    # color 
        (.2, .9),   # roundness  
        (1000, 0),  # poly residual
        (1080, 500),# height
        (200, 800), # arc_length
        (0, 8),     # avg 1st diff
        (10, 1),    # avg 2nd diff 
        (.7, .1),   # avg direction change 
        (600, 200), # contour area 
        (40, 120)]  # lifetime

# Cost assignment constants
MAX_COLOR_COST = 1
MAX_RADIUS_COST = .4
MAX_TRAVEL = 85
COST_WEIGHTS = [1, .2, 0] # [position, radius, color] match weights
MAX_COST = int(1000 * sum(COST_WEIGHTS)) # Maximum cost for an object pairing used in munkres assignment

# Kalman Filter constants:
DELTA_T = .4 # Time step (Small changes have large effects that confuse me_
KF_MEASUREMENT_MAT = np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], np.float32) # Describes what state is being measured (x and y pos)
KF_TRANSITION_MAT = np.array([
        [1, 0, DELTA_T, 0, .5 * DELTA_T ** 2, 0], 
        [0, 1, 0, DELTA_T, 0, .5 * DELTA_T ** 2],
        [0, 0, 1, 0, DELTA_T, 0],
        [0, 0, 0, 1, 0, DELTA_T],
        [0, 0, 0, 0, 0, 0], # All zeroes to avoid simulating x acceleration
        [0, 0, 0, 0, 0, 1]], np.float32) # describes the system for how pos, vel, acc interct

# Setup a logger to be used within this file and all clients of vb_lib
logger = logging.getLogger('vb_logger');
logger.setLevel(logging.DEBUG) # Minimum level logger will output (debug means everything is logged)

class ContourTrack(object):
    ''' Tracks the current and past states of a contour and predicts future paths

        properties:
           lifetime (int): counts how long the contour has been tracked
           unfound_strikes (int): counts how many frames the contour has not been assigned
           cnt_list (nparray(<x, y>, <x, y>, ...)): list of contours in this track (end is newest)
           rad_list (int): list of the radii of the min enclosing circle for cnt
           pos_list (<x, y>): list of the center points of the min enclosing circle for cnt
           color_profile_list ([ColorProfile]): list of color profiles within contours
           cnt_color (<h, s, v>): random color used to visually identify the contour between frames
           kf (cv2.KalmanFilter): uses measurements and system model to filter noise and predict the next state
           fit_coeffs ([float, float, float]): the coefficients of the polynomial fit calculated in is_ball

        behaviors:
            update - updates the track with a new measurement as new ContourTrack
            draw - draws the contour, enclosing circle, measured path, predicted path, velocity, and acceleration
                draw_contour - draws the most recent contour
                draw_path - draws the measure path of the contour
                draw_vel_acc - draws the velocity and acceleration
                draw_polynomial_fit - draws the curve of the polynomial fit to the measure positions if is_ball has been called
            predict - returns a prediction for the state in the next frame
            is_ball - returns the confidence that the contour is the ball 
            is_trackable - returns whether the contour can still be tracked (i.e. hasn't been missing too often)
            concatenated_with - returns a new ContourTrack of this track concatenated with another track
            sub_track - returns a new ContourTrack that is a subsection of this one
            rect_contour_mask - returns a mask (monochrome image) for the rectangle enclosing the contour on any given frame
            circle_contour_mask - returns a mask (monochrome image) for the circle enclosing the contour on any given frame

    '''
    DISCARD_PREDICTIONS = 4 # Number of predictions from Kalman Filter to ignore while it is warming up

    ''' Constructor function:
        (Would be a constructor but method overloading is hard in python)
        image (nparray([hue, sat, val])): HSV image used to generate color profile
        pos (<x, y>): The center point of the bounding circle of the detected contour
        rad (int): The radius of the bounding circle
        cnt (nparray(<x, y>, ...)): The detected contour
        birth_frame_num (int): The number of the first frame the object was seen in
    '''
    @staticmethod
    def new_contour(image, pos, rad, cnt, birth_frame_num):
        return ContourTrack(birth_frame_num, 
                birth_frame_num, 
                0, 
                [cnt], 
                [rad], 
                [pos], 
                [ColorProfile(image, cnt)], 
                rand_hsv_color(),
                image.shape)

    ''' Actual constructor:
        Arguments are self-explanitory
    '''
    def __init__(self, birth_frame_num, last_seen_frame, unfound_strikes, cnt_list, rad_list, pos_list, color_profile_list, cnt_color, image_dim):
        self.birth_frame_num = birth_frame_num
        self.last_seen_frame = last_seen_frame
        self.unfound_strikes = unfound_strikes
        self.cnt_list = cnt_list
        self.rad_list = rad_list
        self.pos_list = pos_list
        self.pre_list = []
        self.color_profile_list = color_profile_list
        self.cnt_color = cnt_color
        self.image_dim = image_dim
        self.fit_coeffs = None

    ''' Creates a new ContourTrack that begins with this track and ends with other
        other (ContourTrack): The track to concatenate on the end of this one
        Returns (ContourTrack): this + other
    '''
    def concatenated_with(self, other):
        # TODO interpolate over gaps and remove overlap
        return ContourTrack(self.birth_frame_num,
                other.last_seen_frame,
                self.unfound_strikes + other.unfound_strikes,  
                self.cnt_list + other.cnt_list,
                self.rad_list + other.rad_list,
                self.pos_list + other.pos_list,
                self.color_profile_list + other.color_profile_list,
                self.cnt_color,
                self.image_dim)

    ''' Creates a new ContourTrack that is a subsection of this one
        start (int): frame number of the frame to start the subsection in
        end (int): last frame to include in the subsection
        Returns (ContourTrack): new contour track with same color, and a subset of the measurements of this one
    '''
    def sub_track(self, start, end):
        return ContourTrack(start,
                end,
                0, # is 0 Okay?
                self.cnt_list[start - self.birth_frame_num:end - self.birth_frame_num],
                self.rad_list[start - self.birth_frame_num:end - self.birth_frame_num],
                self.pos_list[start - self.birth_frame_num:end - self.birth_frame_num],
                self.color_profile_list[start - self.birth_frame_num:end - self.birth_frame_num],
                self.cnt_color,
                self.image_dim)

    ''' Predicts the next state (x, y, x_vel, y_vel, x_acc, y_acc) of the contour
        Returns (<x, y>): Position tuple of the next state
    '''
    def predict(self):
        if len(self.pre_list) <= ContourTrack.DISCARD_PREDICTIONS:
            return self.pos_list[-1]
        pred = self.kf.predict()
        return int(pred[0]), int(pred[1])

    ''' Updates the path, predictions, and state of the contour
        new_track (ContourTrack): The newly created contour track that is the next state of the contour
            new_track (if specified) must come from the same image as self
            A value of None means that the contour was not found in a given frame and does not update the kalman filter with a measured value.
    '''
    def update(self, frame_num, new_track=None):
        # TODO prediction behavior has been commented because it is not helpful, remove it or use it
        # Updates state if a new_track was found, otherwise it updates the track with predicted values
        if new_track:
            self.last_seen_frame = frame_num
            self.pos_list.append(new_track.get_pos())
            self.rad_list.append(new_track.get_rad())
            self.cnt_list.append(new_track.get_cnt())
            self.color_profile_list.append(new_track.get_profile())
            #mea = np.array(new_track.get_pos(), np.float32)
            #self.kf.correct(mea)
        else:
            self.unfound_strikes += 1
            self.pos_list.append(self.pos_list[-1])
            self.rad_list.append(self.rad_list[-1])
            self.cnt_list.append(self.cnt_list[-1])
            self.color_profile_list.append(self.color_profile_list[-1])
            #pre = self.kf.predict()
            #mea = np.array((pre[0], pre[1]), np.float32)
            #self.kf.correct(mea)
        #pre = self.kf.predict()
        #self.pre_list.append(pre)

    ''' Computes the confidence that the contour being tracked is the ball based on many features
        Uses linear combination of normalized fit values for all factors weighted by BALL_WEIGHTS
        Returns (float): Number between 0 and 1 of confidence self is a track of the ball
    '''
    def is_ball(self):
        features = []
        logger.debug('Calculating confidence for %r' % self)

        # Calculates consistency of radius change (balls moving relative to camera will grow and shrink consistently)
        rad_diff_list = []
        last = None
        for rad in self.rad_list:
            if last:
                rad_diff_list.append(abs(rad - last))
            last = rad
        features.append(np.std(rad_diff_list))
        logger.debug('std of raduis differences: %f' % features[-1])

        # Calculate the color_fit as the average profile match between frames
        color_matches = 0
        last_profile = None
        for profile in self.color_profile_list:
            if last_profile:
                color_matches += profile.matches(last_profile)
            last_profile = profile
        features.append(color_matches / (len(self.color_profile_list) - 1))
        logger.debug('average color match: %f' % features[-1])

        # Calculate circular fit as average ratio of contour area to circle area
        contour_area = 0
        for contour in self.cnt_list:
            contour_area += cv2.contourArea(contour)
        circle_area = 0
        for radius in self.rad_list:
            circle_area += radius**2
        circle_area *= math.pi
        features.append(contour_area / circle_area)
        logger.debug('avg roundness: %f' % features[-1])

        # Calculate the fit for the path of the ball on a parabola
        x_vals = []
        y_vals = []
        for pos in self.pos_list:
            x_vals.append(pos[0])
            y_vals.append(pos[1])
        coeff, residual, _, _, _ = np.polyfit(x_vals, y_vals, 2, full=True)
        coeff = list(reversed(coeff))
        # Remember coefficients in an instance variable for drawing
        self.fit_coeffs = coeff
        logger.debug('poly coeffs: ' + str(coeff))
        if len(residual) == 0:
            return 0
        if len(coeff) != 3:
            return 0
        if coeff[2] < MIN_POLY_COEFF or coeff[2] > MAX_POLY_COEFF:
            return 0
        features.append(residual[0] / len(x_vals))
        logger.debug('poly residual: %f' % features[-1])

        # Calculate max height of ball
        features.append(float(min(y_vals)))
        logger.debug('height: %f' % features[-1])

        # Calculate arc length, 1st difference of position (velocity), and 2nd difference (acceleration)
        arc_len = 0
        dist_diff_list = []
        last_distance = None
        last = None
        for pos in self.pos_list:
            if last is not None:
                distance = dist(last, pos)
                arc_len += distance
                if last_distance is not None:
                    dist_diff_list.append(abs(distance - last_distance))
                last_distance = distance
            last = pos
        features.append(arc_len)
        logger.debug('arc length %f' % features[-1])
        features.append(arc_len / (len(self.pos_list) - 1))
        logger.debug('average of 1st difference of position %f' % features[-1])
        features.append(sum(dist_diff_list) / len(dist_diff_list))
        logger.debug('average of 2nd difference of position %f' % features[-1])

        # Calculate average change in direction of velocity
        direction_change = 0
        last_angle = None
        last_pos = None
        last_vel = None
        for pos in self.pos_list:
            if pos is last_pos:
                continue
            if last_pos is not None:
                angle = math.atan2(last_pos[1] - pos[1], last_pos[0] - pos[0])
                if last_angle is not None:
                    direction_change += abs(angle - last_angle)
                last_angle = angle
            last_pos = pos
        features.append(direction_change / (len(self.pos_list) - 2))
        logger.debug('avg direction change: %f' % features[-1])

        # Calculate average contour area
        features.append(contour_area / len(self.cnt_list))
        logger.debug('avg area %f' % features[-1])

        # Calculate lifetime
        features.append(float(self.get_lifetime()))
        logger.debug('lifetime %f' % features[-1])

        # Normalize features using BALL_FEATURE_BOUNDS (0 is closer to 1st bounds, 1 is closer to right bound)
        # Bounds create a step function (Everything outside bounds is given a 0 or 1)
        for idx in range(len(features)):
            f = features[idx]
            b = BALL_FEATURE_BOUNDS[idx]
            if b[0] > b[1]:
                features[idx] = 1 - (np.clip(f, b[1], b[0]) - b[1]) / (b[0] - b[1])
            else:
                features[idx] = (np.clip(f, b[0], b[1]) - b[0]) / (b[1] - b[0])
        logger.debug('normalized features: %s' % features)

        # Calculate final confidence using BALL_WEIGHTS normalized between 0 and 1
        confidence = np.dot(features, BALL_WEIGHTS) / sum(BALL_WEIGHTS)
        logger.debug('%s | confidence: %f' % (self, confidence))
        return confidence

    # Uses lifetime and unfound_strikes to determine wether contour is good enough to keep tracking
    def is_trackable(self):
        return self.unfound_strikes <= MAX_UNFOUND_STRIKES and 1 - (self.unfound_strikes / float(self.get_lifetime())) > MIN_FOUND_RATIO

    # Unused
    def mean_shift(self, image):
        last_pos = self.pos_list[-1]
        pos = get_mean(image)
        while last_pos is not pos:
            pos, last_pos = self.get_mean(image, last_pos), pos
        self.update(pos)

    # Unused        
    def get_mean(self, imag, pos=None):
        if pos is None:
            pos = self.pos_list[-1]
        radius = self.rad_list[-1]

        x, y = pos
        x_sum = 0
        y_sum = 0
        total = 0
        for i in range(-radius, radius + 1): # TODO I don't like adding 1 to radius
            if x + i < 0 or x + i >= len(frame):
                continue
            for j in range(-radius, radius + 1):
                if y + j < 0 or y + j >= len(frame[0]):
                    continue
                if i ** 2 + j ** 2 <= radius ** 2:
                    weight = self.color_profile.get_freq(image[x + i, y + j])
                    # TODO weight from bivariate normal distribution pdf
                    x_sum += weight * (x + i)
                    y_sum += weight * (y + j)
                    total += weight

        return (x_sum / total, y_sum / total)

    # Draws the contour and enclosing circle on frame
    def draw_contour(self, frame):
        color = HSV_COLOR['YELLOW']
        if self.unfound_strikes > 0:
            color = HSV_COLOR['RED']
        #cv2.circle(frame, self.pos_list[-1], self.rad_list[-1], color, 3)
        cv2.drawContours(frame, [self.cnt_list[-1]], 0, self.cnt_color, 2)

    # Draws the measured and predicted paths of the particle as connected lines on frame
    def draw_path(self, frame):
        last = self.pos_list[0]
        for point in self.pos_list[1:]:
            cv2.line(frame, point, last, self.cnt_color, 3)
            last = point

        if len(self.pre_list) > ContourTrack.DISCARD_PREDICTIONS:
            last = None
            for pre in self.pre_list[ContourTrack.DISCARD_PREDICTIONS:]:
                point = (int(pre[0]), int(pre[1]))
                if last is not None:
                    cv2.line(frame, point, last, HSV_COLOR['BLACK'], 3)
                last = point

    # Draws the velocity and acceleration vectors coming out of the contour onto frame
    def draw_vel_acc(self, frame):
        vel_x = int(self.pre_list[-1][2])
        vel_y = int(self.pre_list[-1][3])
        cv2.line(frame, self.pos_list[-1], (self.pos_list[-1][0] + 2 * vel_x, self.pos_list[-1][1] + 2 * vel_y), HSV_COLOR['ORANGE'], 3)

        acc_x = 2 * int(self.pre_list[-1][4])
        acc_y = 2 * int(self.pre_list[-1][5])
        cv2.line(frame, self.pos_list[-1], (self.pos_list[-1][0] + acc_x, self.pos_list[-1][1] + acc_y), HSV_COLOR['CYAN'], 3)

    def draw_polynomial_fit(self, frame, num_points): 
        if self.fit_coeffs is None:
            return
        x_vals = []
        for pos in self.pos_list:
            x_vals.append(pos[0])
        last = None
        for x in np.linspace(min(x_vals), max(x_vals), num_points):
            point = (int(x), int(np.polynomial.polynomial.polyval(x, self.fit_coeffs)))
            if point[0] < 0 or point[0] >= frame.shape[1] or point[1] < 0 or point[1] >= frame.shape[0]:
                continue
            if last is not None:
                cv2.line(frame, last, point, self.cnt_color, 3)
            last = point

    # Draws the contour, path, velocity vector, and acceleration vector onto frame
    def draw(self, frame):
        self.draw_contour(frame)
        self.draw_path(frame)
        self.draw_polynomial_fit(frame, 50)
        self.draw_vel_acc(frame)

    ''' Generates a mask around the tracked contour at a certain frame
        frame_num (int): The frame number corresponding to the contour to be used
        padding (int): The area around the circle to capture aswell 
        Returns (2d grayscale image): The image with the pixels around the contour as 1s and the rest as 0s
    '''
    def circle_contour_mask(self, frame_num, padding):
        mask = np.zeros((self.image_dim[0], self.image_dim[1]), np.uint8)
        if frame_num < self.birth_frame_num or frame_num > self.last_seen_frame:
            return mask
        idx = frame_num - self.birth_frame_num
        cv2.circle(mask, self.pos_list[idx], self.rad_list[idx] + padding, 255, -1)
        return mask

    ''' Generates a mask around the tracked contour at a certain frame
        frame_num (int): The frame number corresponding to the contour to be used
        padding (int): The additional width and height on both sides to make the rectangle
        Returns (2d grayscale image): The image with the pixels around the contour as 1s and the rest as 0s
    '''
    def rect_contour_mask(self, frame_num, padding):
        mask = np.zeros((self.image_dim[0], self.image_dim[1]), np.uint8)
        if frame_num < self.birth_frame_num or frame_num > self.last_seen_frame:
            return mask

        x, y, w, h = cv2.boundingRect(self.cnt_list[frame_num - self.birth_frame_num])
        if x + w + padding > self.image_dim[1]:
            w = self.image_dim[1] - x
        w += padding
        if y + h + padding > self.image_dim[0]:
            h = self,image_dim[0] - y
        h += padding
        x -= padding
        if x < 0:
            x = 0
        y -= padding
        if y < 0:
            y = 0
        mask[y:y+h, x:x+w] = 255
        return mask

    def get_birth_frame(self):
        return self.birth_frame_num
    
    def get_lifetime(self):
        return self.last_seen_frame - self.birth_frame_num + 1

    def get_last_seen_frame(self):
        return self.last_seen_frame

    def get_cnt(self):
        return self.cnt_list[-1]

    def get_first_cnt(self):
        return self.cnt_list[0]

    def get_rad(self):
        return self.rad_list[-1]

    def get_first_rad(self):
        return self.rad_list[0]

    def get_pos(self):
        return self.pos_list[-1]

    def get_first_pos(self):
        return self.pos_list[0]

    def get_profile(self):
        return self.color_profile_list[-1]

    def get_first_profile(self):
        return self.color_profile_list[0]

    def __str__(self):
        return 'ball(%d - %d)' % (self.birth_frame_num, self.last_seen_frame)

    def __repr__(self):
        return self.__str__()


class ColorProfile(object):
    ''' Describes the freqiencies with which colors appear in a certain region of an image
        properties:
            freq (dict(<h,s,v> => float)): Maps between colors and their relative frequencies
            total_pixels (int): The area of the region the profile describes
            contour (nparray(<x, y>, ...)): The region that the profile describes
        behaviors:
            difference: Finds the difference between two color profiles
            draw: Draws the contour for the color profile
            get_freq: Gets the frequency that a specific color occurs in in profile
    '''
    HUE_BIN_SIZE = 2 # Size of Hue bin (Currently on hue is being binned)
    DETECT_BUFFER = 2 # Buffer around the outside of the contour to discard because edge colors may not be as reliable
    
    ''' Constructor scans the image around the contour and generates a color profile
        image (nparray(<h,s,v>, ...)): The image to find color profile within
        contour (nparray(<x, y>, ...)): The region in the image to find the color profile of
    '''
    def __init__(self, image, contour):
        self.freq = dict()
        self.total_pixels = 0
        self.contour = contour

        # Loops through every pixel in the rectangle bounding the contour
        x, y, w, h = cv2.boundingRect(contour)
        for i in range(0, w):
            if x + i < 0 or x + i >= len(image):
                continue
            for j in range(0, h):
                if y + j < 0 or y + j >= len(image[0]):
                    continue
                # If the pixel is within the contour and the buffer, freq of the binned color is incremented
                if cv2.pointPolygonTest(contour, (x + i, y + j), True) >= self.DETECT_BUFFER:
                    self.total_pixels += 1
                    color = ColorProfile.bin_color(image[y + j][x + i])
                    if color in self.freq:
                        self.freq[color] += 1
                    else:
                        self.freq[color] = 1

        # Totals are normalized by dividing by the total pixels
        for color in self.freq.keys():
            self.freq[color] = self.freq[color] / float(self.total_pixels)

    ''' Finds matchingness between two color profiles of contours between frames by subtracting the two contours and dividing by their combined area
        other (ColorProfile): The color profile to compare self with

        Returns (float): The sum of the common area divided by both profiles areas
    '''
    def matches(self, other):
        match = 0
        for color in self.freq:
            if color in other.freq:
                match += min(self.freq[color], other.freq[color])
        return match / 2

    def difference(self, other):
        return 1 - self.matches(other)

    # Draws the contour onto frame
    def draw(self, frame):
        cv2.drawContours(frame, [self.contour], 0, (0, 0, 255), 3)

    # Gets the frequency of a color in the profile
    def get_freq(self, color):
        if color in self.freq:
            return self.freq[color]
        else:
            return 0

    # Creates an new grayscale image where each pixel corresponds to the commonness of the pixel in the color profile
    def freq_image(self, image):
        result = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        for x in range(320, 1600):
            for y in range(180, 900):
                color = ColorProfile.bin_color(image[y][x])
                if color in self.freq:
                    result[y][x] = int(256 * self.freq[color])
        return result

    def __str__(self):
        s = ''
        for color, freq in sorted(self.freq.iteritems(), key=lambda pair: -pair[1]):
            s += str(color) + ': ' + str(int(freq * 1000) / 10.0) + '\n'
        return s

    def __repr__(self):
        return self.__str__()

    ''' Bins colors to group similiar colors. Puts all saturations and values in 1 bin
        color (<h,s,v>): The color to bin

        Returns (<h,s,v>): The binned color
    '''
    @staticmethod
    def bin_color(color):
        return ColorProfile.HUE_BIN_SIZE * int(color[0] / ColorProfile.HUE_BIN_SIZE), 255, 255


class VideoReader(object):
    ''' Video reader opens and reads a video file as well as writes images to the corresponding output location.
        Abstracts away the low level opencv details used by client programs like color_detect.py
        properties:
            src (str): The name (without file extension) of the video to be read from res folder
            fgbg (BackgroundSubtractor): Used to generate a fgmask of moving objects and background image of static objects
            reading (bool): True if a video is currently being read, False otherwise
            length (int): The number of frames in the src video
            count (int): The frame number of the frame currently being read
            skip (int): Determines how many frames will be skipped (1 = no skip)
            read_sk (bool): Determines whether skvideo vreader or opencv ViderCapture is used 
        behaviors:
            read: Generator function that yields each frame in the video as an hsv image
            write_image: Writes an image out to proper output folder
            read_image: Reades an image that was saved earlier to the out folder
            dump_obj: Pickles an object to the proper output folder
            read_obj: Reads a pickled object from the proper output folder
            obj_exists: Checks if a pickled object exists in the proper output folder
            get_backgrond: Gets the background image found using fgbg
    '''
    DEFAULT_VIDEO_FORMAT = '.mp4'

    ''' Constructor
        src (str): The name (without file extension) of the video to be read from res folder
    '''
    def __init__(self, src, sk_read=False):
        self.src = src
        self.fgbg = cv2.createBackgroundSubtractorMOG2()
        self.reading = False
        self.length = 0
        self.count = 0
        self.skip = 1
        self.sk_read = sk_read

        # Create necessary output folders
        if not os.path.exists('out/' + self.src):
            os.mkdir('out/' + self.src)

        self.setup_logger()

    ''' Generator function that yields each frame, frame number, and foreground mask in the video
        skip (int): Optional parameter that can be used to skip more frames (A value of 3 reads every 3rd frame, skipping 2)
        Yields: frame_number (int), frame (2D HSV img), fgmask (2D grayscale img)
    '''
    def read(self, skip=1, get_bg=True):
        self.reading = True
        self.skip = skip
        logger.info('reading video: ' + self.src + VideoReader.DEFAULT_VIDEO_FORMAT)

        if self.read_sk:
            for x in self.read_sk(skip, get_bg):
                yield x
        else:
            # Open the capture for reading frames
            cap = cv2.VideoCapture('res/' + self.src + VideoReader.DEFAULT_VIDEO_FORMAT)

            # Run through the video, yielding each frame and doing background subtraction
            self.count = 0
            while True:
                success, frame = cap.read()
                if not success:
                    break
                self.count += 1
                if self.count % skip is not 0:
                    continue
                fgmask = None 
                if get_bg:
                    fgmask = bg_subtract(frame, self.fgbg)
                if self.count / skip is 1:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                yield self.count / skip, frame, fgmask
                logger.info('-------------------------------------%d' % (self.count / skip))

        self.reading = False
        self.length = self.count

    def read_sk(self, skip, get_bg):
        # Open the capture for reading frames
        cap = skvideo.io.vreader('res/' + self.src + VideoReader.DEFAULT_VIDEO_FORMAT)

        logger.info('reading video: ' + self.src + VideoReader.DEFAULT_VIDEO_FORMAT)

        # Run through the video, yielding each frame and doing background subtraction
        self.count = 0
        for frame in cap:
            self.count += 1
            if self.count % skip is not 0:
                continue
            fgmask = None 
            if get_bg:
                fgmask = bg_subtract(frame, self.fgbg)
            if self.count / skip is 1:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            yield self.count / skip, frame, fgmask
            logger.info('-------------------------------------%d' % (self.count / skip))

    # Sets up the global variable logger with a console handler and an appropriate file handler
    def setup_logger(self):
        # TODO one logger does not support processing multiple videos in parallel
        global logger
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        #f = open('out/' + self.src + '/vb.log', 'w+')
        #f.close()
        fh = logging.FileHandler('out/' + self.src + '/vb.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    ''' Writes an image to the correct location corresponding with the src video
        image (2d HSV image): The image to write out
        name (str): The name of the image or subfolder it belongs to (folder when video being read, name otherwise)
    '''
    def write_image(self, image, name):
        if self.reading:
            if not os.path.exists('out/' + self.src + '/' + name):
                os.mkdir('out/' + self.src + '/' + name)
            name += '/frame-' + str(self.count / self.skip)
        logger.debug('writing image ' + 'out/' + self.src + '/' + name + '.png')
        if len(image.shape) is 3:
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        cv2.imwrite('out/' + self.src + '/' + name + '.png', image)

    ''' Reads an image from the out folder
        name (str): The name of the image without the file extension
        Returns (numpy.ndarray): The image with the correct name in HSV
    '''
    def read_image(self, name):
        if self.reading:
            name += '/frame-' + str(self.count / self.skip)
        logger.info('reading image ' + 'out/' + self.src + '/' + name + '.png')
        image = cv2.imread('out/' + self.src + '/' + name + '.png', cv2.IMREAD_COLOR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Dumps a picked object
    def dump_obj(self, obj, name):
        logger.info('dumping object ' + 'out/' + self.src + '/' + name + '.p')
        pickle.dump(obj, open('out/' + self.src + '/' + name + '.p', 'wb'))

    # Loads a previously pickled object
    def load_obj(self, name):
        logger.info('loading object ' + 'out/' + self.src + '/' + name + '.p')
        return pickle.load(open('out/' + self.src + '/' + name + '.p', 'rb'))

    # Checks if a previously pickled object exists for this video in out
    def obj_exists(self, name):
        return os.path.isfile('out/' + self.src + '/' + name + '.p')

    # Returns the background image computed using opencv MOG2 background subtractor
    def get_background(self):
        bg = self.fgbg.getBackgroundImage()
        if self.sk_read:
            bg = cv2.cvtColor(bg, cv2.COLOR_RGB2HSV)
        else:
            bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        return bg

    def get_length(self):
        return self.length


''' Implementation of Munkres assignment algorithm. Optimally assigns n tasks to m workers. Created using https://en.wikipedia.org/wiki/Hungarian_algorithm as reference
    cost_mat ([[float, float, ...], [float, ...], ...]): Costs for assigning row to column. A cost of -1 will not be assigned

    Returns ([<row, col>, <row, col>, ...]): Pairs of assignments between a worker and a task
'''
def munkres_assignment(cost_mat):
    cost_mat = copy.deepcopy(cost_mat) # Copies the cost_mat to avoid modifying client version
    rows = len(cost_mat) # number of rows in matrix used throughout
    cols = len(cost_mat[0]) # ^^
    rotated = False # True if the cost_mat was rotated in order to have more cols than rows

    if rows is 1 and cols is 1:
        return [(0, 0)]

    # Uses the transpose of the cost_mat to ensure more cols than rows
    if rows > cols:
        (rows, cols) = (cols, rows)
        rot = [[0 for _ in range(cols)] for _ in range(rows)]
        for row in range(rows):
            for col in range(cols):
                rot[row][col] = cost_mat[col][row]
        cost_mat = rot
        rotated = True

    # Creates dummy rows if unequal number of tasks and workers
    dummy_num = cols - rows
    for i in range(dummy_num):
        cost_mat.append([0] * cols)
    rows = cols

    orig_mat = copy.deepcopy(cost_mat)

    # Matrix elements with cost -1 set to MAX_COST so they won't be assigned
    for row in range(rows):
        for col in range(cols):
            if cost_mat[row][col] is -1:
                cost_mat[row][col] = MAX_COST

    # Subtract the minimum in each row from every element in that row.
    for row in range(rows):
        m = min(cost_mat[row])
        for col in range(cols):
            cost_mat[row][col] -= m
    # Repeat with coloumns
    for col in range(cols):
        m = -1
        for row in range(rows):
            if m is -1 or cost_mat[row][col] < m:
                m = cost_mat[row][col]
        for row in range(rows):
            cost_mat[row][col] -= m
    while True:
        # Do a naive drawing of lines to cover all zeroes in cost_mat
        row_open = [True] * rows
        col_open = [True] * cols
        marked = []
        for row in range(rows):
            for col in range(cols):
                if cost_mat[row][col] is 0 and col_open[col]:
                    marked.append((row, col))
                    col_open[col] = False
                    row_open[row] = False
                    break
        # Return if number of lines is already optimal
        if len(marked) is cols:
            marked = [pos for pos in marked if pos[0] < rows - dummy_num and orig_mat[pos[0]][pos[1]] is not -1]
            if rotated:
                marked = [(col, row) for (row, col) in marked]
            return marked

        # Find optimal layout of lines to cover all zeroes
        row_marked = [False] * rows
        col_marked = [False] * cols
        row_to_mark = row_open
        while True in row_to_mark:
            col_to_mark = [False] * cols
            for row in range(rows):
                if not row_to_mark[row]:
                    continue
                row_marked[row] = True
                for col in range(cols):
                    if cost_mat[row][col] is 0 and not col_marked[col]:
                        col_to_mark[col] = True
            row_to_mark = [False] * rows
            for col in range(cols):
                if not col_to_mark[col]:
                    continue
                col_marked[col] = True
                for row in range(rows):
                    if (row, col) in marked:
                        row_to_mark[row] = True

        row_marked = [not x for x in row_marked]

        # Check if minimum number of lines is assignable yet
        num_marked = 0
        for mark in row_marked:
            if mark:
                num_marked += 1
        for mark in col_marked:
            if mark:
                num_marked += 1
        if num_marked is cols:
            # Find assignemt in each row and col using hopcraft karp and return answer
            chosen = get_unique_zeros(cost_mat)

            chosen = [pos for pos in chosen if pos[0] < rows - dummy_num and orig_mat[pos[0]][pos[1]] is not -1]
            if rotated:
                chosen = [(row, col) for (col, row) in chosen]
            return chosen

        # Since there are too few lines through all zeroes, create more zeroes
        m = -1
        for row in range(rows):
            if row_marked[row]:
                continue
            for col in range(cols):
                if col_marked[col]:
                    continue
                if m is -1 or cost_mat[row][col] < m:
                    m = cost_mat[row][col]

        for row in range(rows):
            for col in range(cols):
                if not col_marked[col]:
                    cost_mat[row][col] -= m
                if row_marked[row]:
                    cost_mat[row][col] += m


''' Hopcraft-karp algorithm implementation to match elements in a bipartite graph
    cost_mat (2d list of floats): matrix representation of a bipartite graph where 0 represents an edge connecting nodes, represented by rows and columns

    Returns ([<row, col>, ...]): List of all matched edges (from row node to col node)
'''
def get_unique_zeros(cost_mat):
    rows = len(cost_mat)
    cols = len(cost_mat[0])

    # Do a niave assignment of zeroes using DFS
    row_open = [True] * rows
    col_open = [True] * cols
    marked = []
    for row in range(rows):
        for col in range(cols):
            if cost_mat[row][col] is 0 and col_open[col]:
                marked.append((row, col))
                col_open[col] = False
                row_open[row] = False
                break
    # TODO Not sure is this loops enough times.
    # Works when only one row is unassigned, but more iterations may be necessary.
    # (Vast majority of test data can be processed with easy assignment)
    for row in range(rows):
        if not row_open[row]:
            continue
        # Do a breadth first search starting at every row node to find reachable columns
        path = []
        for col in range(cols):
            if cost_mat[row][col] is 0:
                path.append((row, col))
                break

        dead_end = []
        while len(path) and not col_open[path[-1][1]]:
            nxt = None
            if len(path) % 2 is 0:
                for col in range(cols):
                    loc = (path[-1][0], col)
                    if loc not in path and loc not in dead_end and \
                            loc not in marked and cost_mat[path[-1][0]][col] is 0:
                        nxt = loc
                        break
            else:
                for r in range(rows):
                    loc = (r, path[-1][1])
                    if loc not in path and loc not in dead_end and loc in marked:
                        nxt = loc
                        break
            if nxt is None:
                dead_end.append(path.pop())
            else:
                path.append(nxt)

        # Go through the paths found by BFS, toggling wether an edge is matched.
        for edge in path:
            if edge in marked:
                marked.remove(edge)
                row_open[edge[0]] = True
                col_open[edge[1]] = True
            elif edge not in marked:
                marked.append(edge)
                row_open[edge[0]] = False
                col_open[edge[1]] = False
    return marked


# Gets a mask of pixels within hsv range in frame
def hsv_thresh(frame):
    HSV_LOWER_BOUND = (17, 0, 0)
    HSV_UPPER_BOUND = (32, 255, 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color = cv2.inRange(frame_hsv, HSV_LOWER_BOUND, HSV_UPPER_BOUND)
    ret, color_thresh = cv2.threshold(color, 1, 255, 0)
    color_thresh = cv2.dilate(color_thresh, kernel)
    color_thresh = cv2.dilate(color_thresh, kernel)
    color_thresh = cv2.medianBlur(color_thresh, 5)
    return color_thresh


# Initializes a Kalman filter that tracks position, velocity, and acceleration in 2D
def init_kalman_filter():
    kf = cv2.KalmanFilter(6, 2)
    kf.measurementMatrix = KF_MEASUREMENT_MAT
    kf.transitionMatrix = KF_TRANSITION_MAT
    # kf.processNoiseCov = np.eye(6)
    return kf


# Uses opencv MOG2 background subtraction to generate a mask of moving foreground elements
def bg_subtract(frame, fgbg):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = fgbg.apply(frame)
    ret, fgmask = cv2.threshold(fgmask, 250, 255, 0)
    
    # Erosions, dilations, and median blur used to eliminate noisy single pixels and smooth contours
    fgmask = cv2.erode(fgmask, kernel)
    fgmask = cv2.erode(fgmask, kernel)
    fgmask = cv2.erode(fgmask, kernel)
    fgmask = cv2.dilate(fgmask, kernel)
    fgmask = cv2.dilate(fgmask, kernel)
    fgmask = cv2.dilate(fgmask, kernel)
    fgmask = cv2.dilate(fgmask, kernel)
    fgmask = cv2.dilate(fgmask, kernel)
    fgmask = cv2.medianBlur(fgmask, 5)
    fgmask = cv2.medianBlur(fgmask, 5)

    return fgmask


''' Finds the min enclosing circle for every contour in contours
    contours ([nparray(<x, y>, ...), ...]): The contours to fit circles around

    Returns ([<roundness, contour, <x, y>, radius>, ...]): A sorted list of tuples containing information about the fitting circles
        (Sorted by % of circle that is filled by the contour)
'''
def fitting_circles(contours):
    roundness = []
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > MIN_CONTOUR_AREA:
            (pos, rad) = cv2.minEnclosingCircle(c)
            x = int(pos[0])
            y = int(pos[1])
            # cv2.drawContours(frame, [hull], 0, (0, 255, 0), 3)
            circ_area = rad ** 2 * math.pi
            roundness.append(((circ_area - area) / circ_area, c, (x, y), int(rad)))
    roundness = sorted(roundness, key=lambda r: r[0])
    return roundness


# Finds the length of a - b. Works on vectors (tuples) of any dimension
def dist(a, b):
    if len(a) is not len(b):
        return False
    length = 0
    for i in range(len(a)):
        length += (a[i] - b[i]) ** 2
    return math.sqrt(length)


''' Cost function used to match contours on similarity in position, color, and radius
    old_track (ContourTrack): The contour that is being tracked already
    new_track (ContourTrack): A newly found contour that can be paired with old_track given an optimal cost

    Returns (float): A weighted combination of position cost, color cost, and radius cost.
'''
def cost(old_track, new_track, max_trav=MAX_TRAVEL):
    pos_cost = dist(old_track.get_pos(), new_track.get_first_pos()) / max_trav
    rad_cost = 1 - float(min(old_track.get_rad(), new_track.get_first_rad())) / max(old_track.get_rad(), new_track.get_first_rad())
    color_cost = old_track.get_profile().difference(new_track.get_first_profile())
    if pos_cost > 1:
        return -1
    if rad_cost > MAX_RADIUS_COST:
        return -1
    if color_cost > MAX_COLOR_COST:
        return -1

    return int(1000 * (COST_WEIGHTS[0] * pos_cost + COST_WEIGHTS[1] * color_cost + COST_WEIGHTS[1] * rad_cost))


# Old function for matching color profiles (replaced in ColorProfile class)
def color_match(image, freq, pos, rad):
    freq = freq.copy()
    total_pixels = 0
    matching = 0
    (x, y) = pos
    for i in range(-rad, rad + 1):
        if x + i < 0 or x + i >= len(image[0]):
            continue
        for j in range(-rad, rad + 1):
            if y + j < 0 or y + j >= len(image):
                continue
            if i ** 2 + j ** 2 <= rad ** 2:
                color = bin_color(image[y + j][x + i])
                total_pixels += 1
                if color in freq and freq[color] > 0:
                    freq[color] -= 1
                    matching += 1
    return matching / float(total_pixels)


# Creates a random hsv color (3-tuple) from uniform distribution spanning entire color space
def rand_hsv_color():
    return (int(np.random.rand() * 180), int(np.random.rand() * 255), int(np.random.rand() * 255))


''' Function is a generator, yields each frame of the video with foreground mask
    src (str): The name (without the extension) of the video to read out of the res folder
    ext (str): The file name extension (i.e. '.mp4', '.mov')
    skip (int): Determines what frames will be read. 1 means every frame, 3 means every 3rd frame

    Yields (<frame number, frame, foreground mask>): Information about the current frame
'''
def read_video(src, ext, skip):
    # Replaced by VideoReader class
    logger.info('reading video %s.%s' % (src, ext))

    cap = skvideo.io.VideoCapture('res/' + src + ext)
    if not os.path.exists('out/' + src):
        os.mkdir('out/' + src)
    if not os.path.exists('out/' + src + '/processed/'):
        os.mkdir('out/' + src + '/processed/')

    count = 0
    success = True
    fgbg = cv2.createBackgroundSubtractorMOG2()
    while count < 100:
        success, frame = cap.read()
        if not success:
            break
        count += 1
        if count % skip is not 0:
            continue

        fgmask = bg_subtract(frame, fgbg)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        yield count / skip, frame, fgmask
        logger.info('-------------------------------------%d' % count / skip)

    background = fgbg.getBackgroundImage()
    cv2.imwrite('out/' + src + '/bg.png', cv2.cvtColor(background, cv2.COLOR_RGB2BGR))
    cap.release()


''' Selects contours based on position above net_line and roundness and creates ContourTracks from them if selected
    frame_hsv (np.ndarray): hsv image contours were found in
    contours ([np.ndarray]): list of contours to be chosen from
    num_to_select (int): maximum number of contours to be selected
    frame_num (int): the number of the frame contours were found in
    net_line ((x, y, slope)): the line representing to net

    Returns ([ContourTrack]): ContourTrack objects for the selected contours
'''
def select_contours(frame_hsv, contours, num_to_select, frame_num, net_line=None):
    roundness = fitting_circles(contours)
    new_objs = []
    for i in range(min(num_to_select, len(roundness))):
        r = roundness[i]
        pos = r[2]
        # Remove contours below the net
        if net_line is not None:
            # TODO Why is this a greater than sign?
            if pos[1] > net_line[0][1] + net_line[2] * (pos[0] - net_line[0][0]):
                continue
        new_objs.append(ContourTrack.new_contour(frame_hsv, (int(r[2][0]), int(r[2][1])), r[3], r[1], frame_num))

    return new_objs


''' Uses munkres assignment algorith to match up new contours to old contour tracks
    objs ([ContourTrack]): Tracks existing before this frame
    new_objs ([ContourTrack]): Contours found in this frame
    frame_num (int): Current frame number

    Return ([ContourTrack]): List of tracks where untrackable tracks are removed, new tracks are added, and paired tracks are updated
'''
def pair_contours(objs, new_objs, frame_num):
    if objs:
        if new_objs:
            # Create a new matrix where rows are old tracks, cols are new tracks, and elements are costs to pair corresponding tracks
            cost_mat = [[0 for _ in new_objs] for _ in objs]
            for i in range(len(objs)):
                for j in range(len(new_objs)):
                    cost_mat[i][j] = cost(objs[i], new_objs[j])
            solutions = munkres_assignment(cost_mat)
            assigned_new = [False] * len(new_objs)
            assigned_old = [False] * len(objs)
            for (old_idx, new_idx) in solutions:
                if new_idx < len(new_objs) and old_idx < len(objs):
                    objs[old_idx].update(frame_num, new_objs[new_idx])
                    assigned_new[new_idx] = True
                    assigned_old[old_idx] = True
            # Create new objects for unassigned new tracks
            for new_idx in range(len(new_objs)):
                if not assigned_new[new_idx]:
                    objs.append(new_objs[new_idx])
            # Give a strike to unassigned old tracks
            for old_idx in range(len(cost_mat)):
                idx = len(cost_mat) - 1 - old_idx
                if not assigned_old[idx]:
                    objs[idx].update(frame_num)
        else:
            objs = []
    else:
        objs = new_objs

    return objs


# unused - trim ball candidates to the part that gives them the highest confidence
def trim_ball_candidates(ball_candidates):
    idx = len(ball_candidates) - 1
    while idx >= 0:
        candidate = ball_candidates[idx]
        trimmed = (0, None)
        for frame_num in range(candidate[1].get_birth_frame() + 1, candidate[1].get_last_seen_frame() - 1):
            if frame_num - candidate[1].get_birth_frame() > candidate[1].get_last_seen_frame() - frame_num:
                a = candidate[1].sub_track(candidate[1].get_birth_frame(), frame_num)
                a = (a.is_ball(), a)
                if a[0] > trimmed[0]:
                    trimmed = a
            else:
                b = candidate[1].sub_track(frame_num, candidate[1].get_last_seen_frame())
                b = (b.is_ball(), b)
                if b[0] > trimmed[0]:
                    trimmed = b
        if trimmed[1] is not None:
            logger.info('trimming %s to %s' % (candidate[1], trimmed[1]))
            ball_candidates[idx] = trimmed
        idx -= 1


# Split ball candidates if it leads to higher confidences
def split_ball_candidates(ball_candidates):
    SPLIT_TOLERANCE = .05
    idx = len(ball_candidates) - 1
    while idx >= 0:
        candidate = ball_candidates[idx]
        best_a = (candidate[0] + SPLIT_TOLERANCE, None)
        best_b = (candidate[0] + SPLIT_TOLERANCE, None)
        for frame_num in range(candidate[1].get_birth_frame() + 3, candidate[1].get_last_seen_frame() - 3):
            logger.debug('examining splitting %r at frame %d' % (candidate, frame_num))
            a = candidate[1].sub_track(candidate[1].get_birth_frame(), frame_num)
            b = candidate[1].sub_track(frame_num, candidate[1].get_last_seen_frame())
            a = (a.is_ball(), a)
            b = (b.is_ball(), b)
            if a[0] + b[0] > best_a[0] + best_b[0]:
                best_a = a
                best_b = b
        if best_a[1] is not None:
            logger.info('Splitting %s at %d' % (candidate[1], best_b[1].get_birth_frame()))
            ball_candidates[idx] = best_b
            ball_candidates.insert(idx, best_a)
        idx -= 1


# Concatenate ball candidates if that leads to higher confidence
def concatenate_ball_candidates(ball_candidates):
    MAX_PAIR_OVERLAP = 3
    MAX_PAIR_GAP = 15
    CONCATENATION_TOLERANCE = .15
    MAX_CONCATENATION_COST = 500

    # Selects pairs of candidates A and B, concatenates if makes better ball candidate than both A and B
    # Does the above in a way that concatenated pairs will also be considered for concatenation
    # allowing for every possible chain of concatenation to be considered

    ball_candidates.sort(key=lambda c: c[1].get_birth_frame())
    idxA = len(ball_candidates) - 1
    while idxA >= 0:
        candidateA = ball_candidates[idxA]
        idxB = idxA + 1
        while idxB < len(ball_candidates):
            candidateB = ball_candidates[idxB]
            idxB += 1
            if candidateA[1].get_last_seen_frame() - candidateB[1].get_birth_frame() > MAX_PAIR_OVERLAP:
                continue
            if candidateB[1].get_birth_frame() - candidateA[1].get_last_seen_frame() > MAX_PAIR_GAP:
                break

            c = cost(candidateA[1], candidateB[1], 150)
            logger.debug('checking concatenation of %r with %r' % (candidateA[1], candidateB[1]))
            logger.debug('cost = %f' % c)
            if c is not -1 and c < MAX_CONCATENATION_COST:
                concatenated = candidateA[1].concatenated_with(candidateB[1])
                confidence = concatenated.is_ball()
                if confidence + CONCATENATION_TOLERANCE < (candidateA[0] + candidateB[0]) / 2:
                    continue
                logger.info('concatenating %r with %r' % (candidateA[1], candidateB[1]))
                ball_candidates.pop(idxB - 1)
                ball_candidates[idxA] = (confidence, concatenated)
                break
        idxA -= 1
    # TODO make sure that selected candidates are not sub chains of eachother
    # Possibly by removing candidates that have been concatenated into better candidates.
    # ^^ Would that cause issues?
