import cv2 as cv
import streamlit as st
import numpy as np
from PIL import Image
from PIL import ImageOps
import set_solver.find as find
import time

def solve(img):
    return find.find_sets(img)

def main_loop():
    st.title("Set Solver")
    st.write('Note: an assertion error will trigger if the script runs for more than 25 seconds. Streamlit seems to lose connection when this happens...')
    st.write('NB .HEIC images will take longer (even if it is not listed as .HEIC when uploaded).')
    upload = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])
    img = None
    if upload is not None:
        file_bytes = np.asarray(bytearray(upload.read()), dtype=np.uint8)
        opencv_image = cv.imdecode(file_bytes, 1)
        img = cv.cvtColor(opencv_image, cv.COLOR_BGR2RGB)
    else:
        return None
    
    start = time.time()
    solutions, labelings = solve(img)
    end = time.time()
    st.header("Total Execution Time: "+str(end - start))
    st.header("Card labels")
    st.image(labelings)
    if len(solutions) == 0:
        st.header("There are no sets")
    else:
        st.header("There are "+str(len(solutions))+" sets" if len(solutions) >= 2 else "There is 1 set")
        for solution in solutions:
            st.image(solution)
    
        
if __name__ == '__main__':
    main_loop()
