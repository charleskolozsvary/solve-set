import cv2 as cv
import numpy as np
import math
import statistics
from copy import deepcopy
import streamlit as st

BLUR_SIZE = 7
GAUSSIAN_DEV = 1000
BINARY_CAP = 165 #needs to be well-lit environment ¯\_(ツ)_/¯ (or does it?)
QUADRANGLE_TOLERANCE = .05 #0.1
CARDSIZE_TOLERANCE_UPPER = 2 #2.5
CARDSIZE_TOLERANCE_LOWER = 1/2 #2/5
NUM_CARDS_FOR_MEDIAN_HIGH = 15
NUM_CARDS_FOR_MEDIAN_LOW = 4
NUM_MAX_CARDS = 100 #in case of duplicates, which will be dealt with in find.py

R_1 = [(0, 0, 165), (180, 65, 255)]
R_2 = [(0, 0, 65), (180, 65, 255)]
R_3 = [(0, 0, 0), (180, 125, 255)]


CARD_EXTRACT_RANGE_BRIGHT = [(0, 0, 150), (180, 75, 255)]
CARD_EXTRACT_RANGE_DARK = [(0, 0, 70), (180, 75, 255)]
CARD_EXTRACT_RANGE_SAT_MED = [(0, 0, 120), (180, 120, 255)]
CARD_EXTRACT_RANGE_SAT_LOW = [(0, 0, 120), (180, 90, 255)]
SAT_RANGE_EXTREME_LOW = [(0, 0, 120), (180, 50, 255)]
SAT_RANGE_EXTREME_LOW_DARK = [(0, 0, 100), (180, 50, 255)]
SAT_RANGE_LOWEST_BRIGHT = [(0, 0, 190), (180, 30, 255)]

RANGES = [CARD_EXTRACT_RANGE_BRIGHT, SAT_RANGE_LOWEST_BRIGHT, CARD_EXTRACT_RANGE_DARK, CARD_EXTRACT_RANGE_SAT_MED, CARD_EXTRACT_RANGE_SAT_LOW, SAT_RANGE_EXTREME_LOW, SAT_RANGE_EXTREME_LOW_DARK, R_1, R_2, R_3]
        
def inRadius(candidate_centroid, equiv_class_centroid):
    for value in equiv_class_centroid:
        c = value[0]
        radius = value[1]
        if np.sqrt((candidate_centroid[0] - c[0])**2 + (candidate_centroid[1] - c[1])**2) <= radius:
            return True
    return False

def uniqueContours(contours, centers):
    contour_partition = {}
    centroid_partition = {} #equivalence classes
    size_of_partition = 0
    for i, cnt in enumerate(contours):
        centroid = centers[i]
        epsilon = QUADRANGLE_TOLERANCE*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        _, width, height = standardize(np.reshape(approx, (4, 2)))
        radius = min(width, height) / 2
        count = 0
        for key in contour_partition:
            if inRadius(centroid, centroid_partition[key]):
                contour_partition[key].append(cnt) #'''deepcopy'''
                centroid_partition[key].append( (centroid, radius) )
                count += 1
        if count == 0:
            contour_partition[size_of_partition] = [cnt] #'''deepcopy'''
            centroid_partition[size_of_partition] = [(centroid, radius)]
            size_of_partition += 1
        
    unique_contours = []
    #get representatives, choosing contours with the least number of points
    for key, equiv_class in contour_partition.items():
        largestContourArea = max(list(map(lambda x : cv.contourArea(x), equiv_class)))
        def func(x):
            area = cv.contourArea(x)
            return area < largestContourArea * 2.25 and area > largestContourArea * 0.35
        largest_within_max = sorted(list(filter(func, equiv_class)), key = cv.contourArea, reverse = True)[:max(len(RANGES), 3)]
        smallest_by_contour_len = sorted(largest_within_max, key = len)
        representative_contour = smallest_by_contour_len[0].copy() #smallest_by_contour_len# #'''deepcopy'''
        unique_contours.append(representative_contour) #pick the contour with the least number of points
        
        
    return unique_contours

def filterContours(orig, contours):
    quads = []
    for cnt in contours:
        epsilon = QUADRANGLE_TOLERANCE*cv.arcLength(cnt, True)
        approx = cv.approxPolyDP(cnt, epsilon, True)
        if len(approx) == 4:
            quads.append(cnt)
    largestQuads = sorted(quads, key = cv.contourArea, reverse = True)
    intermediate = list(map(lambda x : cv.contourArea(x), largestQuads))
    largestAreas1 = intermediate[:NUM_CARDS_FOR_MEDIAN_HIGH]
    largestAreas2 = intermediate[:NUM_CARDS_FOR_MEDIAN_LOW] #'''deepcopy'''
    
    if len(largestAreas1) == 0:
        return []
    med1 = statistics.median(largestAreas1)
    med2 = statistics.median(largestAreas2)
    med = med2 if med1 < 0.2 * med2 else med1
    output = list(filter(lambda x: cv.contourArea(x) < CARDSIZE_TOLERANCE_UPPER * med and cv.contourArea(x) > CARDSIZE_TOLERANCE_LOWER * med, largestQuads))
    return output[:NUM_MAX_CARDS]
        
def thresholdAndFilterContours(hsv_img, ranges, extra_threshs):
    allContours = []
    allCenters = []
    threshs = []
    blurred = cv.GaussianBlur(hsv_img, (BLUR_SIZE, BLUR_SIZE), GAUSSIAN_DEV, GAUSSIAN_DEV)
    for range in ranges:
        threshs.append(cv.inRange(blurred, range[0], range[1]))
    for thresh in extra_threshs:
        threshs.append(thresh)
    
    for thresh in threshs:
#        st.image(thresh)
        conts, _ = cv.findContours(thresh, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
        filteredContours = filterContours(hsv_img, conts)
        for cnt in filteredContours:
            M = cv.moments(cnt)
            c = (int(M['m10']/M['m00']), int(M['m01']/M['m00']))
            allCenters.append(c)
            allContours.append(cnt)
            
    unique_contours = uniqueContours(allContours, allCenters)
    last_filter = filterContours(hsv_img, unique_contours)

    assert len(last_filter) >= 0, "none of the thresholding ranges produced desirable contours?"
    return last_filter
    
def standardize(quadrangle):
    assert len(quadrangle) == 4, "Houston, we've got a problem: quadrangle is said to be made up of more than 4 points."
    points = []
    for point in quadrangle:
        points.append((point[0], point[1]))
    bottomRight = max(points, key = lambda x : sum(x))
    topLeft = min(points, key = lambda x: sum(x))
    points.remove(bottomRight)
    points.remove(topLeft)
    topRight = max(points, key = lambda x: x[0])
    points.remove(topRight)
    assert len(points) == 1, "Removing more than allowed points?"
    bottomLeft = points[0]
    width = min(math.dist(topLeft, topRight), math.dist(bottomLeft, bottomRight))
    height = min(math.dist(topLeft, bottomLeft), math.dist(topRight, bottomRight))
    return np.float32([topRight, topLeft, bottomLeft, bottomRight]), width, height
    
''' MOST GOING ON HERE '''
def getCardImagesAndTheirContours(orig_img):
    hsv_img = cv.cvtColor(orig_img, cv.COLOR_RGB2HSV)
    gray_img = cv.cvtColor(orig_img, cv.COLOR_RGB2GRAY)
    blurred = cv.GaussianBlur(gray_img, (BLUR_SIZE, BLUR_SIZE), GAUSSIAN_DEV, GAUSSIAN_DEV)
    _, thresh = cv.threshold(blurred, BINARY_CAP, 255, cv.THRESH_BINARY)
    cardContours = thresholdAndFilterContours(hsv_img, RANGES, [thresh])
    cardImages = []
    quadrangles = []
    cardCenters = []
    for cnt in cardContours:
        epsilon = QUADRANGLE_TOLERANCE * cv.arcLength(cnt, True)
        quadrangle = cv.approxPolyDP(cnt, epsilon, True)
        quadrangles.append(quadrangle)
        reshapedQuad = np.reshape(quadrangle, (4, 2))
        def midpoint_pos(p1, p2):
            return (int(.5 * p1[0] + 0.5 * p2[0]), int(0.5 * p1[1] + 0.5 * p2[1]))
        corners, width, height = standardize(reshapedQuad)
        cardCenters.append(midpoint_pos(corners[1], corners[2]))
        Mat = cv.getPerspectiveTransform(corners, np.float32([[width, 0], [0, 0], [0, height], [width, height]]))
        extractedCard = cv.warpPerspective(orig_img, Mat, (int(width), int(height)), flags = cv.INTER_LINEAR)
        cardImages.append(extractedCard)
            
    return cardImages, quadrangles, cardCenters
    

