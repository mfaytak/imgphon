from imutils import face_utils
from PIL import Image, ImageDraw
import numpy as np
import imutils
import dlib
import cv2
import os, subprocess, csv, glob
import matplotlib.pyplot as plt

def detect_landmarks(my_ndarray):
    """
    Input: an ndarray frame output from cv2.VideoCapture object.
    Output: a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    """

    # read in image TODO change to something more general like the commented-out line
    gray = cv2.cvtColor(my_ndarray, cv2.COLOR_BGR2GRAY)
    # gray = np.asarray(cv2image, dtype=np.uint8)
    
    # TODO cheekpad obliteration happens here if remove_cheekpad=True
    
    # run face detector to get bounding rectangle
    # TODO pass detector object into function
    rect = detector(gray, 1)[0]
    
    # run landmark prediction on portion of image in face rectangle; output
    # TODO pass predictor object into function
    shape = predictor(gray, rect)
    shape_np = face_utils.shape_to_np(shape)
    
    return shape_np
    
def draw_landmarks(cv2_video_capture, shape, aperture_xy = False):
    """
    Inputs: a cv2.VideoCapture object (TODO change), and a (68,2) ndarray of x,y coords that dlib detects.
    Outputs: an image with lines drawn over the detected landmarks; useful for testing and visualization.
    aperture_xy: if True, also draw (next to face) numerical values for x and y diameters of lip aperture.
    """

    out_image = cv2_video_capture.copy()

    for i,name in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
        if name == "mouth":
            continue
        j,k = face_utils.FACIAL_LANDMARKS_IDXS[name]
        pts = np.array(shape[j:k], dtype=np.uint32)
        for idx,pt in enumerate(pts):
            pt1 = pt
            try:
                pt2 = pts[idx+1]
            except IndexError:
                if name == "left_eye" or name == "right_eye":
                    pt2 = pts[0]
                else:
                    continue
            cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255))
    
    # drawing the mouth with some more precision
    # draw most of the outer perimeter of lips
    jm,km = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    for idx in range(jm,jm+11): 
        pt1 = shape[idx]
        pt2 = shape[idx+1]
        cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255))
    
    # draw the last segment for the outer perimiter of lips
    cv2.line(out_image, tuple(shape[48]), tuple(shape[59]), (255,255,255))
    
    # draw the inner aperture of the lips
    for idx in range(jm+12,km):
        pt1 = shape[idx]
        try:
            pt2 = shape[idx+1]
        except IndexError:
            pt2 = shape[jm+12]
        cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255))
        
    # add text indicating measured lip aperture in px
    if aperture_xy:
        x,y = get_lip_aperture(shape)
        add_string = "x={}, y={}".format(round(x,1),round(y,1))
        loc = tuple(np.subtract(shape[4], (200,0)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_image, add_string, loc, font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
    return out_image

def get_lip_aperture(shape):
    """
    Inputs: the typical 68,2 ndarray "shape" object output by detect_landmarks.
    Outputs: a 2-tuple of horizontal and vertical diameters of the lip aperture, 
     treating the horizontal line like the major axis of an ellipse,
     and the vertical line like the minor axis.
    """
    horizontal_axis = np.linalg.norm(shape[60] - shape[64])
    vertical_axis = np.linalg.norm(shape[62] - shape[66])

    return horizontal_axis,vertical_axis
    
