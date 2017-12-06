from imutils import face_utils
from PIL import Image, ImageDraw
from scipy import ndimage
import numpy as np
import imutils
import dlib
import cv2
import os, subprocess, csv, glob
import matplotlib.pyplot as plt

# instantiate dlib face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def get_video_frame(video, time):
    """
    Return the single frame closest to the given timepoint. Can then run detect_landmarks on the frame.
    Inputs: video - an MXF file; time in seconds - format d.ddd (sec.msec), rounded to three decimal places.
    Outputs: an ndarray image of the desired frame.
    """
    output_bmp = 'test3.bmp'
    try:
        os.remove(output_bmp)
    except OSError:
        pass
    frame_get_args = ['ffmpeg', '-i', video, 
                      '-vcodec', 'bmp', 
                      '-ss', time,
                      '-vframes', '1', 
                      '-an', '-f', 'rawvideo',
                       output_bmp]
    subprocess.check_call(frame_get_args)
    frame = cv2.imread(output_bmp)
    return frame

def detect_landmarks(my_ndarray, detector=detector, predictor=predictor):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, 
            a detector of choice from dlib,
            and a dlib face landmark predictor trained on data of choice.
    Output: a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    """

    # read in image TODO change to something more general like the commented-out line
    gray = cv2.cvtColor(my_ndarray, cv2.COLOR_BGR2GRAY)
    # gray = np.asarray(cv2image, dtype=np.uint8)
    
    # run face detector to get bounding rectangle
    rect = detector(gray, 1)[0]
    
    # run landmark prediction on portion of image in face rectangle; output
    shape = predictor(gray, rect)
    shape_np = face_utils.shape_to_np(shape)
    
    return shape_np
    
def draw_landmarks(my_ndarray, shape, aperture_xy = False):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, and a (68,2) ndarray of x,y coords that dlib detects.
    Outputs: an image with lines drawn over the detected landmarks; useful for testing and visualization.
    aperture_xy: if True, also draw (next to face) numerical values for x and y diameters of lip aperture.
    """

    out_image = my_ndarray.copy()

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
