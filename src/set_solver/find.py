'''
Given basic card information, find the sets
'''
import set_solver.extract as extract
import set_solver.identify as identify
import cv2 as cv
import itertools
import streamlit as st
import numpy as np
import time

DISPLAY_SET_COLOR = (25, 75, 225)
LABEL_COLOR = (50, 120, 225)
CONTOUR_WIDTH_PROP = 0.0003
TEXT_SCALING_PROP = 0.0000125
MIN_CONTOUR_DRAW_WIDTH = 12
MIN_TEXT_SIZE = 1
MAX_DISPLAY_TIME = 5

FEATURES = [['R', 'G', 'P'], ['RH', 'OV', 'SQ'], ['1', '2', '3'], ['F', 'E', 'D']]

def getComplementCard(c1, c2):
    complement = []
    card1, card2 = c1.split(':'), c2.split(':')
    for i in range(4):
        if card1[i] == card2[i]:
            complement.append(card1[i])
        else:
            options = list(FEATURES[i])
            options.remove(card1[i])
            options.remove(card2[i])
            complement.append(options[0])
    return ':'.join([complement[0], complement[1], complement[2], complement[3]])
    
def getSets(cards): #cards is dict with key of strings representing cards and the value as the index which the card corresponds to in the
    sets = []
    for pair in itertools.combinations(cards.keys(), 2):
        comp = getComplementCard(pair[0], pair[1])
        if comp in cards.keys():
            set = frozenset([cards[pair[0]], cards[pair[1]], cards[comp]])
            if set not in sets:
                sets.append(set)
    return sets
    
    
def displaySets(originalImage, sets, cardContours, cardCenters, labels):
    labeledImg = originalImage.copy()
    s1 = time.time()
    drawing_widths = []
    for i, label in enumerate(labels):
        cnt = cardContours[i]
        area = cv.contourArea(cnt)
        drawing_widths.append(max(MIN_CONTOUR_DRAW_WIDTH, int(CONTOUR_WIDTH_PROP * area)))
        text_size = max(MIN_TEXT_SIZE, area * TEXT_SCALING_PROP)
        cv.putText(labeledImg, label, cardCenters[i], cv.FONT_HERSHEY_SIMPLEX, text_size, LABEL_COLOR, int(text_size * 3), 2)
    total_time = time.time() - s1
    solutions = []
    for set in sets:
        assert total_time < MAX_DISPLAY_TIME, 'It has taken longer than '+str(MAX_DISPLAY_TIME)+' seconds to display the sets...'
        start = time.time()
        solution = originalImage.copy()
        for index in set:
            cv.drawContours(solution, [cardContours[index]], 0, DISPLAY_SET_COLOR, drawing_widths[index])
        solutions.append(solution)
        total_time += time.time() - start
    return solutions, labeledImg
    
def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0]) ** 2 + (p1[1]-p2[1]) ** 2)

def remove_insert(arr, old, new, idx):
    if type(old) == type(np.array([], dtype = 'int32')):
        for i, cnt in enumerate(arr):
            if (cnt == old).all():
                del arr[idx]
                arr.insert(idx, new.copy())
                break
        else:
            st.write('old contour was not present')
    else:
        arr.remove(old)
        arr.insert(idx, new)
    
def removeDuplicateCards(contours, positions, labels, center_of_img, imgs, orig_img):
    cards = {}
    unique_contours, unique_positions = [], []
    card_idx = 0
    for i, label in enumerate(labels):
        key = ':'.join([label[0][0], label[1][:2], label[2][0], label[3][0]]).upper()
        if key in cards.keys():
            st.write('duplicate handeled...')
            if dist(positions[i], center_of_img) < dist(unique_positions[cards[key]], center_of_img):
                remove_insert(unique_contours, unique_contours[cards[key]], contours[i], cards[key])
                remove_insert(unique_positions, unique_positions[cards[key]], positions[i], cards[key])
            continue
        else:
            cards[key] = card_idx
            unique_contours.append(contours[i].copy())
            unique_positions.append(positions[i])
            card_idx += 1
    return cards, unique_contours, unique_positions
    
    
def find_sets(orig_img): #img is in RGB
    s1 = time.time()
    cardImages, cardContours, cardCenters = extract.getCardImagesAndTheirContours(orig_img)
    e1 = time.time()
    t1 = e1 - s1
    st.write('Time spent extracting cards:', t1)
    s2 = time.time()
    cardLabels = identify.getCardLabels(cardImages)
    e2 = time.time()
    t2 = e2 - s2
    st.write('Time spent identifying cards:', t2)
    cards, unique_contours, unique_positions = removeDuplicateCards(cardContours, cardCenters, cardLabels, (orig_img.shape[0] // 2, orig_img.shape[1] // 2), cardImages, orig_img)
    s4 = time.time()
    sets = getSets(cards)
    e4 = time.time()
    t4 = e4 - s4
    st.write('Time spent finding sets:', t4)
    s5 = time.time()
    solutions, labels = displaySets(orig_img, sets, unique_contours, unique_positions, cards.keys())
    e5 = time.time()
    t5 = e5 - s5
    st.write('Time spent displaying sets:', t5)
    return solutions, labels
