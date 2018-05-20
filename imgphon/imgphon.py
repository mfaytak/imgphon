import os, subprocess, csv, glob
from imutils import face_utils
import numpy as np
import dlib
import cv2
import imutils
from imutils.face_utils import FaceAligner
from imutils.face_utils import rect_to_bb
from skimage.draw import polygon
from scipy.ndimage.measurements import center_of_mass
from scipy.ndimage import zoom

def get_video_frame(video, time):
    """
    Return the single frame closest to the given timepoint. Can then run detect_landmarks on the frame.
    Inputs: video - an MXF file; time in seconds - format d.ddd (sec.msec), rounded to three decimal places.
    Outputs: an ndarray image of the desired frame.
    """
    output_bmp = 'temp.bmp'
    try:
        os.remove(output_bmp)
    except OSError:
        pass
    frame_get_args = ['ffmpeg', 
                      '-nostats',
                      '-loglevel', 'warning',
                      '-ss', str(time),
                      '-i', video, 
                      '-vcodec', 'bmp', 
                      '-vframes', '1', 
                      '-an', '-f', 'rawvideo',
                       output_bmp]
    subprocess.check_call(frame_get_args)
    frame = cv2.imread(output_bmp)
    return frame

def detect_landmarks(frame, detector, predictor):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, 
        a detector of choice from dlib,
        and a dlib face landmark predictor trained on data of choice.
    Outputs:the portion of the image containing the detected face, 
        and a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    """

    # read in image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # run face detector to get bounding rectangle
    rect = detector(gray, 1)[0]
    
    # run landmark prediction on portion of image in face rectangle; output
    shape = predictor(gray, rect)
    marks = face_utils.shape_to_np(shape)
    
    return marks

def get_norm_face(frame, detector, predictor, aligner):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, 
        a detector of choice from dlib,
        a dlib face landmark predictor trained on data of choice,
        and an imutils FaceAligner instance.
    Outputs:an affine-transformed/rotated and rescaled face bounding box
        and a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    Inspired by code by Adrian Rosebrock.
    """
    # TODO! build in ability to manually define where eye is?
    # for cases where eyes are consistently mis-detected
    # read in image 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # run face detector to get bounding rectangle
    rect = detector(gray,1)[0]

    # align face
    # TODO fork imutils, change this function to rotate and return the landmarks used for alignment; 
    # replace current marks_np
    faceAligned = aligner.align(frame, gray, rect)

    return faceAligned

def anonymize(frame, marks, area = "narrow"):
    """
    Inputs: an image ndarray and a (68,2) ndarray of x,y coords for facial landmarks, such as that
        output by detect_landmarks.
    Outputs: an image with Gaussian blur applied to an area determined by the detected landmarks.
    Options: area (of the blurred region), which can either be "narrow", around the eyes, or 
        "broad", covering the whole upper portion of the face.
    """
    out_image = frame.copy()

    if area == "broad":
        # get bottom of box using nose
        nose_start,nose_end = face_utils.FACIAL_LANDMARKS_IDXS['nose']
        max_y = max([marks[i][1] for i in range(nose_start, nose_end)])

        # get top of box using eyebrows
        eyebrows_start = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow'][0]
        eyebrows_end = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow'][1]
        min_y = min([marks[i][1] for i in range(eyebrows_start, eyebrows_end)])

        # get sides of box using jaw
        jaw_start,jaw_end = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
        max_x = max([marks[i][0] for i in range(jaw_start, jaw_end)])
        min_x = min([marks[i][0] for i in range(jaw_start, jaw_end)])

        # pad out the blurred area
        # TODO change to integer pixel values (will throw error in notebooks)
        width = max_x - min_x
        height = max_y - min_y
        min_y -= 0.4*height
        min_x -= 0.1*width
        max_y += 0.1*height
        max_x += 0.1*width

        # replace the area around the selected landmarks with a blurred version of the area
        upper_face = out_image[min_y:max_y, min_x:max_x]
        blur = cv2.GaussianBlur(upper_face,(101, 101), 30)
        out_image[min_y:min_y+upper_face.shape[0], min_x:min_x+upper_face.shape[1]] = blur

    elif area == "narrow":
        # get bottom of box using eyes
        eyes_start = face_utils.FACIAL_LANDMARKS_IDXS['right_eye'][0]
        eyes_end = face_utils.FACIAL_LANDMARKS_IDXS['left_eye'][1]
        max_y = max([marks[i][1] for i in range(eyes_start, eyes_end)])

        # get top of box using eyebrows
        eyebrows_start = face_utils.FACIAL_LANDMARKS_IDXS['right_eyebrow'][0]
        eyebrows_end = face_utils.FACIAL_LANDMARKS_IDXS['left_eyebrow'][1]
        min_y = min([marks[i][1] for i in range(eyebrows_start, eyebrows_end)])

        # get sides of box using jaw
        jaw_start,jaw_end = face_utils.FACIAL_LANDMARKS_IDXS['jaw']
        max_x = max([marks[i][0] for i in range(jaw_start, jaw_end)])
        min_x = min([marks[i][0] for i in range(jaw_start, jaw_end)])
            
        # pad out bottom boundary
        height = max_y - min_y
        max_y += 0.4*height

        upper_face = out_image[min_y:max_y, min_x:max_x]
        blur = cv2.GaussianBlur(upper_face,(101, 101), 30)
        out_image[min_y:min_y+upper_face.shape[0], min_x:min_x+upper_face.shape[1]] = blur

    else:
        raise RuntimeError('Argument area must be "narrow" or "broad"')

    return out_image
    
def draw_landmarks(frame, marks, anonymize = "none", aperture_xy = False, line_width=2):
    """
    Inputs: an ndarray frame output from cv2.VideoCapture object, and a (68,2) ndarray of x,y coords that dlib detects.
    Outputs: an image with lines drawn over the detected landmarks; useful for testing and visualization.
    anon: apply a Gaussian blur to the top of the face to remove identifiable features, with options
      "none", "broad", or "narrow" (see anonymize function for details).
    aperture_xy: if True, also draw (next to face) numerical values for x and y diameters of lip aperture.
    line_width: adjust thickness of lines connecting the landmarks.
    """
    out_image = frame.copy()
    lwd = int(line_width)

    if anonymize != "none":
        out_image = anonymize(out_image,marks,area=anonymize)

    for i,name in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
        if name == "mouth":
            continue
        j,k = face_utils.FACIAL_LANDMARKS_IDXS[name]
        pts = np.array(marks[j:k], dtype=np.uint32)
        for idx,pt in enumerate(pts):
            pt1 = pt
            try:
                pt2 = pts[idx+1]
            except IndexError:
                if name == "left_eye" or name == "right_eye":
                    pt2 = pts[0]
                else:
                    continue
            cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255),lwd)
    
    # drawing the mouth with some more precision
    # draw most of the outer perimeter of lips
    jm,km = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    for idx in range(jm,jm+11): 
        pt1 = marks[idx]
        pt2 = marks[idx+1]
        cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255),lwd)
    
    # draw the last segment for the outer perimiter of lips
    cv2.line(out_image, tuple(marks[48]), tuple(marks[59]), (255,255,255),lwd)
    
    # draw the inner aperture of the lips
    for idx in range(jm+12,km):
        pt1 = marks[idx]
        try:
            pt2 = marks[idx+1]
        except IndexError:
            pt2 = marks[jm+12]
        cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255),lwd)

    # add text indicating measured lip aperture in px
    if aperture_xy:
        x,y = get_lip_aperture(marks)
        add_string = "x={}, y={}".format(round(x,1),round(y,1))
        loc = tuple(np.subtract(marks[4], (200,0)))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(out_image, add_string, loc, font, 0.8, (255,255,255), 2, cv2.LINE_AA)
        
    return out_image

def get_lip_aperture(marks):
    """
    Inputs: the typical 68,2 ndarray "marks" object output by detect_landmarks.
    Outputs: a 2-tuple of horizontal and vertical diameters of the lip aperture, 
     treating the horizontal line like the major axis of an ellipse,
     and the vertical line like the minor axis.
    """
    horizontal_axis = np.linalg.norm(marks[60] - marks[64])
    vertical_axis = np.linalg.norm(marks[62] - marks[66])

    return horizontal_axis,vertical_axis

# TODO convert lip_mask into a class and add the other clean-up functions below as methods

def lip_mask(frame, marks):
    """
    Returns a simplified ndarray containing 0s/1s, with lips a filled polygon of 1s
    """
    # fetch indices of mouth landmark points
    jm,km = face_utils.FACIAL_LANDMARKS_IDXS['mouth']
    
    # initialize a blank background in the shape of the original image
    mask_dims = [frame.shape[0],frame.shape[1]]
    mask = np.zeros(mask_dims, dtype=np.uint8)

    # fill outer lip polygon
    mouth_outer = marks[jm:jm+11]
    mouth_outer_col = [p[0] for p in mouth_outer]
    mouth_outer_row = [p[1] for p in mouth_outer]
    # last point's coords need to be appended manually
    mouth_outer_col.append(marks[59][0])
    mouth_outer_row.append(marks[59][1])
    rr,cc = polygon(mouth_outer_row,mouth_outer_col)
    mask[rr,cc] = 1

    # then, cancel out inner polygon
    mouth_inner = marks[jm+12:km]
    mouth_inner_col = [p[0] for p in mouth_inner]
    mouth_inner_row = [p[1] for p in mouth_inner]
    rr,cc = polygon(mouth_inner_row,mouth_inner_col)
    mask[rr,cc] = 0

    return mask

def get_center(image):
    """ Returns the center location of an array as (x, y). """
    return np.array([image.shape[0] // 2, image.shape[1] // 2])
  
def centroid(image):
    """ Returns centroid. """
    if (image == 0).all():
        return get_center(image)
    centroid = center_of_mass(image)
    centroid = np.rint(np.array(centroid))
    return centroid.astype(int)

def cross(image, point):
    """ Point-marking (with cross shape) function, to mark off i.e. centroid. """
    image_w, image_h = image.shape
    x, y = point
    blank = np.zeros(image.shape)
    blank[max(0, x - 3):min(x + 4, image_w), y] = 1
    blank[x, max(0, y - 3):min(y + 4, image_h)] = 1
    blank[x,y] = 0
    return blank.astype(bool).astype(int)

#from scipy.ndimage.measurements import center_of_mass
#from scipy.ndimage import zoom

def crop_center(input_mask):
    """ 
    Returns mask (1s/0s) with centroid of 1s centered on center of a smaller ground of 0s. 
    This normalizes for some aspects of head movement (relative size and absolute position),
      but does not normalize for left/right head tilt.
    """
    mask = input_mask.copy()
    # define a smaller ground of zeros
    # TODO make this scaled to the individual acquisition somehow, by e.g. lip width
    ground_shape = np.array([int(d/3) for d in mask.shape])
    ground = np.zeros(ground_shape)
    
    # get mask's centroid and ground's center point
    cg_x, cg_y = get_center(ground)
    mask_centroid = centroid(mask)
    ci_x, ci_y = mask_centroid
    # unpack dims of mask and ground
    image_w, image_h = mask.shape
    ground_w, ground_h = ground_shape
    
    # determine preliminary amounts to crop from left and top
    left_diff = ci_x - cg_x
    top_diff = ci_y - cg_y
    
    # check if left/top sides of image fit on ground; chop off if not
    if left_diff > 0:
        mask = mask[left_diff:,]
        ci_x -= left_diff
    if top_diff > 0:
        mask = mask[:,top_diff:]
        ci_y -= top_diff
    
    # get mask dims again
    image_w, image_h = mask.shape
        
    # get right and bottom dimensions
    ground_r = ground_w - cg_x
    ground_b = ground_h - cg_y
    image_r = image_w - ci_x
    image_b = image_h - ci_y 
    
    # determine amounts to crop from right and bottom
    right_diff = ground_r - image_r
    bottom_diff = ground_b - image_b
    
    # check if right/bottom sides of image fit on ground; chop off if not
    if right_diff < 0:
        mask = mask[:image_w + right_diff]
    if bottom_diff < 0:
        mask = mask[:,:image_h + bottom_diff]
        
    image_w, image_h = mask.shape

    # copy modified mask onto ground shape
    left_start = cg_x - ci_x
    top_start = cg_y - ci_y
    ground[left_start:left_start + image_w, top_start:top_start + image_h] += mask
        
    return ground
