from imutils import face_utils
from PIL import Image, ImageDraw
from scipy import ndimage
import numpy as np
import imutils
import dlib
import cv2
import os, subprocess, csv, glob
import matplotlib.pyplot as plt

class Cheekpad:
    """ Cheekpad object that mediates finding and replacing of cheekpads.
        Intended to be initiated at the beginning of detecting landmarks for 
        one subject if cheekpad replacement is needed.
    """

    def __init__(self, initial_frame):
        self.refPt = []
        self.cp_ROI = []
        self.color_points = []
        self.cp_color_upper = np.empty(3, dtype=int)
        self.cp_color_lower = np.empty(3, dtype=int)
        _ = self.get_params(initial_frame)
        self.avg_color = self.get_avg_color(frame)
        ''' TODO(?) count number of frames done for this subject, 
            prompt manual check of cheekpad-finding accuracy every x frames
        '''

    def get_params(self, frame):
        """ Displays first frame, then defines cp region of interest and color parameters 
            on the basis of mouse clicks. """

        tb_name = "adjust trackbar to mask cheekpads, press q when done"

        def callback(value):
            pass

        def setup_trackbars():
            """ Produces trackbars that can be used to adjust cp_color_lower and cp_color_upper values."""

            cv2.namedWindow(tb_name, 0)
            cv2.resizeWindow(tb_name, 1000, 400)

            j, i = 'R', 'MIN'
            v = self.cp_color_lower[2]
            cv2.createTrackbar("{}_{}".format(j, i), tb_name, v, 255, callback)

        def get_trackbar_values():
            """ Resets cp_color_upper and/or cp_color_lower on basis of trackbar position. """

            self.cp_color_lower = np.zeros(3).astype(int)
            self.cp_color_upper = 255 * np.ones(3).astype(int)

            j, i = 'R', 'MIN'
            v = cv2.getTrackbarPos("{}_{}".format(j, i), tb_name)
            self.cp_color_lower[2] = v

        def mouse_click(event, x, y, flags, param):
            """ Puts coordinate of mouse click in refPt."""

            if event == cv2.EVENT_LBUTTONUP:
                self.refPt.append((x, y))

        window_name = "click around the outer edges of both cheekpads, press q when done"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_click)
        
        while True:  # collect a few points to sample the color and location
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if  key == ord("q"):  # wait for 'q'
                cv2.destroyWindow(window_name)
                break
            
        # get all x and y values of locations clicked
        xlocs = [int(i[0]) for i in self.refPt]
        ylocs = [int(i[1]) for i in self.refPt]

        # cp_color_range is min to max for each pixel in image[self.refPt[i]]
        bgr = frame[[ylocs],[xlocs],][0]     
        for j in range(0,3): # loop over RGB (actually BGR)
            self.cp_color_upper[j] = max([i[j] for i in bgr])
            self.cp_color_lower[j] = min([i[j] for i in bgr])   
                
        # calculate a bounding box - ROI for finding cps - Assume stationary speaker
        minx=0; miny =0; maxx=640; maxy=360
        minx = min(xlocs)
        maxx = max(xlocs)
        miny = min(ylocs)
        maxy = max(ylocs)
        self.cp_ROI.append(int(minx))
        self.cp_ROI.append(int(miny))
        self.cp_ROI.append(int(maxx))
        self.cp_ROI.append(int(maxy))
        
        (mask, output, box, height, width) = self.find_cheekpads(frame)
        
        setup_trackbars()  # the color sliders

        while True:
            get_trackbar_values()
            (mask, output, box, height, width) = self.find_cheekpads(frame)
            cv2.imshow(tb_name, output)  
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        cv2.destroyWindow(tb_name)
        cv2.destroyWindow("Preview")
        return(mask, output, box, height, width)

    def find_cheekpads(self, frame):
        """ Using filter params and cp region of interest, provide height, width, etc of cps. """

        cp_ROI_copy = frame[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]] # isolate cps from frame
        box = np.array([[0,0], [0,1], [1,1], [1,0]])  # initialize variables
        height = 0
        width = 0

        mask = 255 - cv2.inRange(cp_ROI_copy, self.cp_color_lower, self.cp_color_upper) # binary mask, TODO invert for display
        output = cv2.bitwise_and(cp_ROI_copy, cp_ROI_copy, mask = mask) # masked original image.

        # use the contours of the mask to find width and height of found cps
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            height = max(c[...,0,1]) - min(c[...,0,1])
            width = max(c[...,0,0]) - min(c[...,0,0])
        return (mask.astype(bool).astype(int), output, box, height, width) 
            
    def get_avg_color(self, frame):
        """ Gets colors from image selected by user and returns their average. """

        def mouse_click_color(event, x, y, flags, param):
            """ Puts BGR color array at mouse click location in color_points."""

            if event == cv2.EVENT_LBUTTONUP:
                self.color_points.append(frame[y,x])
        
        window_name = "click a couple points on the face to sample color, press q when done"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, mouse_click_color)
        
        while True:
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if  key == ord("q"):  # wait for 'q'
                cv2.destroyWindow(window_name)
                break
        return np.rint(np.average(self.color_points, axis=0))

    def cheekpad_tidy(self, mask):
        """ Input: results of masking on lip image. 
            Output: evened-out, cleaned up image.
        """
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask_temp = 255 * mask.astype('uint8')
        mask_closed = cv2.morphologyEx(mask_temp, cv2.MORPH_CLOSE, se1)
        mask_dilated = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, se2)
        
        # find all blobs
        blobs, number_of_blobs = ndimage.label(mask)
        blob_areas = ndimage.sum(mask, blobs, index=range(blobs.max() + 1))
        area_filter = (blob_areas < 1500)
        blobs[area_filter[blobs]] = 0 # only include blobs with area >= 1500
        
        mask = blobs.astype('bool').astype(int)
        mask_h, mask_w = mask.shape
        x_buf_l = int(round(mask_w / 3))
        x_buf_r = int(round(mask_w * (2/3)))
        new_mask = np.zeros(mask.shape)
        new_mask[:,0:x_buf_l] = mask[:,0:x_buf_l]
        new_mask[:,x_buf_r:] = mask[:,x_buf_r:]
        
        return new_mask

    def replace_cheekpads(self, frame):
        """ Input: frame nd_array, Cheekpad object
            Output: frame with cheekpads replaced
        """
        (mask, output, box, height, width) = self.find_cheekpads(frame)
        cleaned = self.cheekpad_tidy(mask)
        cps = np.copy(frame[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]])
        cps[np.where(cleaned)] = self.avg_color
        frame[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]] = cps
        return frame


def detect_landmarks(my_ndarray, detector=detector, predictor=predictor, cheekpad=None):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, 
            a detector of choice from dlib,
            and a dlib face landmark predictor trained on data of choice.
    Output: a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    """
    # if cheekpad is initialized, replace cheekpads
    if cheekpad:
        my_ndarray = cheekpad.replace_cheekpads(my_ndarray)

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