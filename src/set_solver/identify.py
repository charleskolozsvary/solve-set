## from extract import * ## need to find out what exactly we were using from import...

from sklearn.cluster import KMeans
from statistics import median
import streamlit as st
import numpy as np
import time

import cv2 as cv
import statistics

MAX_TIME = 20

RED_RANGE = [(0, 100, 100), (22, 255, 255)] #[(0, 100, 100), ...]
PURPLE_RANGE = [(115, 0, 0), (175, 255, 255)]
PURPLE_RANGE_TWO = [(0, 0, 0), (180, 255, 70)]
GREEN_RANGE = [(30, 20, 0), (90, 255, 255)]
PERCENTS = [0.085, 0.15, 0.22] #proportion necessary to be filled if one, two, or three is the number, respectively

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

TOTAL_DIFF_TO_BACKROUND_CAP = 27
TOTAL_DIFF_TO_PERIMETER_CAP = -10

MINIMUM_PURPLE_PERCENTAGE = 0.15 #0.1
MINIMUM_GREEN_PERCENTAGE = 1/3

CONTOUR_APPROX_TOLERANCE = 0.011

OUTERMOST_CONTOUR_SIZE_TOLERANCE = 1/3
INNER_CONTOUR_SIZE_TOLERANCE = 1/4
            
def filterShapeContours(outer, inner):
    areas = list(map(lambda x : cv.contourArea(x), outer))
    m = max(areas)
    def tolerance_func(tol, center):
        return lambda x : cv.contourArea(x) > center*(1-tol) and cv.contourArea(x) < (1+tol)*center
    outer_filtered = list(filter(tolerance_func(OUTERMOST_CONTOUR_SIZE_TOLERANCE, m), outer))
    inner_filtered = list(filter(tolerance_func(INNER_CONTOUR_SIZE_TOLERANCE, statistics.median(areas)), inner))
    return outer_filtered, inner_filtered
    
def clustering(cardImg):
#    oddness = cardImg
#    if cardImg.shape[2] == 4:
#        oddness = np.delete(cardImg, 3, 2)
    flattened = np.reshape(cardImg, (cardImg.shape[0] * cardImg.shape[1], 3))
    kmeans = KMeans(n_clusters = 2, random_state = 0, n_init = "auto").fit(flattened)
    blackLabel = 0 if sum(kmeans.cluster_centers_[0]) > sum(kmeans.cluster_centers_[1]) else 1
#    nonWhiteColor = kmeans.cluster_centers_[1 ^ whiteLabel] # indx == 1 if whiteLabel 0, else 1
#    nonWhiteColor = (nonWhiteColor[0], nonWhiteColor[1], nonWhiteColor[2])
    simplified = np.uint8(list(map(lambda x : BLACK if x == blackLabel else WHITE, kmeans.labels_)))
    simp_reshape = np.reshape(simplified, (cardImg.shape[0], cardImg.shape[1], 3))
    return cv.cvtColor(simp_reshape, cv.COLOR_RGB2GRAY)
    
def getContoursAndMasks(cardImg, thresh):
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    #countourHierarchyInfo = [Next, Prev, firstChild, Parent]
    #see https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html for more info
    if hierarchy is None or contours is None:
        return [], [], [], [], []
    outer_contours = []
    inner_contours = []
    for i, contourHierarchyInfo in enumerate(hierarchy[0]):
        parent = contourHierarchyInfo[3]
        if parent == -1:
            outer_contours.append(contours[i])
        else:
            inner_contours.append(contours[i])
    outer_contours, inner_contours = filterShapeContours(outer_contours, inner_contours)
    emptyCard = np.zeros((cardImg.shape[0], cardImg.shape[1]), dtype = 'uint8')
    arc_lengths = list(map(lambda x : cv.arcLength(x, True), outer_contours))
    avgLength = statistics.mean(arc_lengths) if len(outer_contours) != 0 else 0
    shapePerimeterMask = cv.drawContours(emptyCard.copy(), outer_contours, -1, WHITE, 1) #'''deepcopy'''
    shapeFillingMask = cv.drawContours(emptyCard, outer_contours, -1, WHITE, -1) #'''deepcopy''' #-1 for line width arg  draws the contour filled in
    d_size = max(int(0.025 * avgLength), 1)
    kernel_size = (d_size, d_size)
    def dilateAndErode(mask):
        dilated_perim = cv.dilate(mask, cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size))
        return dilated_perim
#        return cv.erode(dilated_perim, cv.getStructuringElement(cv.MORPH_CROSS, kernel_size))

    shapePerimeterMask = dilateAndErode(shapePerimeterMask)
    shapeFillingMask = dilateAndErode(shapeFillingMask)
    return contours, outer_contours, inner_contours, shapePerimeterMask, shapeFillingMask

def labelCard(cardImg): #already rgb image
    thresh = clustering(cardImg)
    n_tot_pixels = cardImg.shape[0] * cardImg.shape[1]
    hsv_img = cv.cvtColor(cardImg, cv.COLOR_RGB2HSV)
    threshRed = cv.inRange(hsv_img, RED_RANGE[0], RED_RANGE[1])
    threshGreen = cv.inRange(hsv_img, GREEN_RANGE[0], GREEN_RANGE[1])
    threshPurple = cv.inRange(hsv_img, PURPLE_RANGE[0], PURPLE_RANGE[1])
    threshPurpleTwo = cv.inRange(hsv_img, PURPLE_RANGE_TWO[0], PURPLE_RANGE_TWO[1])
    threshAllColors = cv.bitwise_or(cv.bitwise_or(cv.bitwise_or(threshRed, threshGreen), threshPurple), threshPurpleTwo)
    
    trad_conts, trad_outer_contours, trad_inner_contours, trad_shapePerimeterMask, trad_shapeFillingMask = getContoursAndMasks(cardImg, thresh)
    new_contours, new_outer_contours, new_inner_contours, new_shapePerimeterMask, new_shapeFillingMask = getContoursAndMasks(cardImg, threshAllColors)
    
    n_trad_outermost = len(trad_outer_contours)
    n_new_outermost = len(new_outer_contours)
    
    contours, outer_contours, inner_contours, shapePerimeterMask, shapeFillingMask = [], [], [], [], []
    
    if new_contours == []:
            contours, outer_contours, inner_contours, shapePerimeterMask, shapeFillingMask = trad_conts, trad_outer_contours, trad_inner_contours, trad_shapePerimeterMask, trad_shapeFillingMask
    else:
        def isValid(val):
            return val <= 3 and val >= 1 #prioritize num_outer_to be within range
        def face_off(first, second): #return true if option1 better
            first_valid = isValid(first[0])
            second_valid = isValid(second[0])
            if first_valid and second_valid:
                if first[0] == second[0]:
                    if first[1] == second[1]:
                        return True #tiebreak to first
                    else:
                        return first[1] <= second[1] #last, prioritize number non-zero in filled mask (less is better)
                else:
                    return first[0] > second[0] #next prioritize those with higher number of outer contours
            else:
                return first_valid
        
        option1 = [n_trad_outermost, cv.countNonZero(trad_shapeFillingMask)]
        option2 = [n_new_outermost, cv.countNonZero(new_shapeFillingMask)]
        chose_trad = face_off(option1, option2)
        
        percentage = cv.countNonZero(new_shapeFillingMask)/n_tot_pixels
        if (isValid(n_new_outermost) and percentage < PERCENTS[n_new_outermost-1]) or cv.countNonZero(new_shapeFillingMask) <= 1.25 * cv.countNonZero(new_shapePerimeterMask): #when new option doesn't fill very much of the card, revert back to previous method
            chose_trad = True
        
        if chose_trad:
            contours, outer_contours, inner_contours, shapePerimeterMask, shapeFillingMask = trad_conts, trad_outer_contours, trad_inner_contours, trad_shapePerimeterMask, trad_shapeFillingMask
            chose_trad = True
        else:
            contours, outer_contours, inner_contours, shapePerimeterMask, shapeFillingMask = new_contours, new_outer_contours, new_inner_contours, new_shapePerimeterMask, new_shapeFillingMask
            
    ################################# NUMBER v v v
    num_outermost = len(outer_contours)
    cardNumber = str(num_outermost)
    assert num_outermost <= 3 and num_outermost >= 1, "card number could not be found for a card"
    ################################## NUMBER ^ ^ ^
    
    ######################################### FILLING v v v
    backgroundMask = cv.bitwise_not(shapeFillingMask)
    grabBackround = cv.bitwise_and(cardImg, cardImg, mask = backgroundMask)
    overlap = cv.bitwise_and(shapeFillingMask, shapePerimeterMask)
    justFilling = cv.bitwise_xor(overlap, shapeFillingMask)
    grabForeground = cv.bitwise_and(cardImg, cardImg, mask = justFilling)
    grabPerim = cv.bitwise_and(cardImg, cardImg, mask = overlap)
    avgBackground = cv.mean(grabBackround, mask = backgroundMask)
    avgPerim = cv.mean(grabPerim, mask = overlap)
    avgForeground = cv.mean(cardImg, mask = justFilling)
    
    def pixel_diff(avg1, avg2):
        total = 0
        for i in range(3):
            total += avg1[i] - avg2[i] #want signed difference, since if foreground is lighter than background, it is unlikely that the filling is dashed.
        return total
    diff_back = pixel_diff(avgBackground, avgForeground) #if the foreground (center of the shape) is ligher than the background, then the shape is almost certainly empty
    diff_perim = pixel_diff(avgForeground, avgPerim) #if foreground is darker than perimeter, it is very likely filled, so that is why we order as we do
    
    cardFilling = ''
    if diff_perim < TOTAL_DIFF_TO_PERIMETER_CAP:
        cardFilling = 'full'
    elif diff_back < TOTAL_DIFF_TO_BACKROUND_CAP:
        cardFilling = 'empty'
    else:
        cardFilling = 'dashed'
    ################################################# FILLING ^ ^ ^
    
    ######################################################### COLOR v v v
    card_HSV = cv.cvtColor(cardImg, cv.COLOR_RGB2HSV)
    color_threshs = []
    for ran in [RED_RANGE, GREEN_RANGE, PURPLE_RANGE, PURPLE_RANGE_TWO]:
        color_threshs.append(cv.inRange(card_HSV, ran[0], ran[1]))
    masked_color_threshs = list(map(lambda x : cv.bitwise_and(x, shapePerimeterMask), color_threshs))
#    masked_color_threshs = list(map(lambda x : cv.bitwise_and(x, shapeFillingMask), color_threshs))
    masked_color_threshs[3] = cv.bitwise_or(masked_color_threshs[2], masked_color_threshs[3])
    
    counts = list(map(cv.countNonZero, masked_color_threshs))
    s = sum([counts[0], counts[1], counts[2]])
    
    cardColor = ''
    if (s != 0 and counts[1]/s > MINIMUM_GREEN_PERCENTAGE) or (counts[1] > 10 and counts[1] / (counts[1] + counts[2]) > 0.7 and counts[3] > 30 and counts[0] < 75):
        cardColor = 'green'
    elif (s - counts[1]) == 0 or s == 0:
        cardColor = 'purple'
    #TODO: If the card is smaller, a smaller count of red should be required
    elif counts[0] <= 250 and (counts[2]/(s - counts[1]) > MINIMUM_PURPLE_PERCENTAGE or (counts[3] >= 3 and counts[0] < 125)): #magic numbers, yes, I know...
        cardColor = 'purple'
    else:
        cardColor = 'red'
    
    ############################## COLOR ^ ^ ^ ^
    
    ########################################## SHAPE v v v
    approxes = []
    for cont in outer_contours:
        epsilon = CONTOUR_APPROX_TOLERANCE * cv.arcLength(cont, True)
        approxes.append(cv.approxPolyDP(cont, epsilon, True))
        
    approxContour = sorted(approxes, key = len)[0]
    cardShape = ''
    if len(approxContour)  <= 7:
        cardShape = 'rhombus'
    elif cv.isContourConvex(approxContour):
        cardShape = 'oval'
    else:
        cardShape = 'squiggle'
    ######################## SHAPE ^ ^ ^
    
    return cardColor, cardFilling, cardNumber, cardShape

def getCardLabels(cardImgs):
    labels = []
    total_time = 0
    for i, cardImg in enumerate(cardImgs):
        assert total_time < MAX_TIME, 'It has taken longer than '+str(MAX_TIME)+' seconds to identify the cards...'
#        st.write(str(total_time))
        start = time.time()
        color, filling, number, shape = labelCard(cardImg)
        labels.append((color, shape, number, filling))
        total_time += time.time() - start
    return labels
