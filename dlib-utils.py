def detect_landmarks(cv2_video_capture):
    """
    Input: a cv2.VideoCapture object (TODO change).
    Output: a (68,2) ndarray containing X,Y coordinates for the 68 face points dlib detects.
    """

    # read in image TODO change to something more general like the commented-out line
    gray = cv2.cvtColor(cv2_video_capture, cv2.COLOR_BGR2GRAY)
    # gray = np.asarray(cv2image, dtype=np.uint8)
    
    # run face detector to get bounding rectangle
    rect = detector(gray, 1)[0]
    
    # run landmark prediction on portion of image in face rectangle; output
    shape = predictor(gray, rect)
    shape_np = face_utils.shape_to_np(shape)
    
    return shape_np
    
def draw_landmarks(cv2_video_capture, shape):
    """
    Inputs: a cv2.VideoCapture object (TODO change), and a (68,2) ndarray of x,y coords that dlib detects.
    Outputs: an image with lines drawn over the detected landmarks; useful for testing and visualization.
    """

    out_image = cv2_video_capture.copy()

    for i,name in enumerate(face_utils.FACIAL_LANDMARKS_IDXS.keys()):
        j,k = face_utils.FACIAL_LANDMARKS_IDXS[name]
        pts = np.array(shape[j:k], dtype=np.uint32)
        for idx,pt in enumerate(pts):
            pt1 = pt
            try:
                pt2 = pts[idx+1]
            except IndexError:
                if name == "mouth" or name == "left_eye" or name == "right_eye":
                    pt2 = pts[0]
                else:
                    continue
            cv2.line(out_image, tuple(pt1), tuple(pt2), (255,255,255))
        
    return out_image
    
