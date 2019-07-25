'''
Functions and classes which define an image segmentation-based approach to "removing" an ultrasound
helmet from frontal images. Intended for use along with facial landmark detection utilities (./landmark.py).
'''

import cv2
import numpy as np
import os

from scipy import ndimage

class CheekpadSegment:
    """ CheekpadSegment object that mediates finding and removing of cheekpads.
        Intended to be initiated at the beginning of detecting landmarks for 
        one subject if cheekpad replacement is needed.
    """

    def __init__(self, frame):
        self.refPt = []
        self.cp_ROI = []
        self.color_points = []
        self.cp_color_upper = np.empty(3, dtype=int)
        self.cp_color_lower = np.empty(3, dtype=int)
        _ = self.get_params(frame)
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

        window_name = "click OUTSIDE the outer edges of both cheekpads, press q when done"
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
        
        (mask, output) = self.find_cheekpads(frame)
        
        setup_trackbars()  # the color sliders

        while True:
            get_trackbar_values()
            (mask, output) = self.find_cheekpads(frame)
            output_h, output_w, _ = output.shape
            output_copy = np.copy(output)
            output_copy[:, output_w // 3] = 255
            output_copy[:, output_w * 2 // 3] = 255
            cv2.imshow(tb_name, output_copy)  
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        cv2.destroyWindow(tb_name)
        cv2.destroyWindow("Preview")
        return (mask, output)

    def find_cheekpads(self, frame):
        """ Using filter params and cheekpad_segment region of interest, 
            returns the cheekpad mask and masked output of the frame.
        """

        cp_ROI_copy = frame[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]] # isolate cps from frame

        mask = 255 - cv2.inRange(cp_ROI_copy, self.cp_color_lower, self.cp_color_upper) # binary mask, TODO invert for display
        output = cv2.bitwise_and(cp_ROI_copy, cp_ROI_copy, mask=mask) # masked original image.

        return (mask.astype(bool).astype(int), output)
            
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
        # set up structuring elements
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        #se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
        kernel = np.ones((4,4),np.uint8)

        # close
        mask_temp = 255 * mask.astype('uint8')
        mask_closed = cv2.morphologyEx(mask_temp, cv2.MORPH_CLOSE, se)
        
        # find all blobs and remove the small ones (speckles)
        blobs, number_of_blobs = ndimage.label(mask_closed)
        blob_areas = ndimage.sum(mask_closed, blobs, index=range(blobs.max() + 1))
        area_filter = (blob_areas < 1500)
        blobs[area_filter[blobs]] = 0 # only include blobs with area >= 1500

        # dilate
        mask_temp = 255 * blobs.astype('uint8')
        mask_dilated = cv2.dilate(mask_temp, kernel, iterations=4)

        # close again
        mask_final = cv2.morphologyEx(mask_dilated, cv2.MORPH_CLOSE, se)
        
        # undo middle third of mask
        mask_final = mask_final.astype('bool').astype(int)
        mask_h, mask_w = mask_final.shape
        x_buf_l = int(round(mask_w / 3))
        x_buf_r = int(round(mask_w * (2/3)))
        out_mask = np.zeros(mask_final.shape)
        out_mask[:,0:x_buf_l] = mask_final[:,0:x_buf_l]
        out_mask[:,x_buf_r:] = mask_final[:,x_buf_r:]

        # finally remove any remaining small blobs and return
        blobs, number_of_blobs = ndimage.label(out_mask)
        blob_areas = ndimage.sum(out_mask, blobs, index=range(blobs.max() + 1))
        area_filter = (blob_areas < 2000)
        blobs[area_filter[blobs]] = 0 # only include blobs with area >= 2000

        out_mask = blobs.astype('bool').astype(int)
        
        return out_mask

    def remove(self, frame):
        """ Input: frame nd_array
            Output: frame with cheekpads removed
        """
        frame_copy = np.copy(frame)
        (mask, output) = self.find_cheekpads(frame_copy)
        cleaned_mask = self.cheekpad_tidy(mask)
        cps = np.copy(frame_copy[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]])
        cps[np.where(cleaned_mask)] = self.avg_color
        frame_copy[self.cp_ROI[1]:self.cp_ROI[3], self.cp_ROI[0]:self.cp_ROI[2]] = cps
        if self.check_adjust(frame_copy):
            self.get_params(frame)
            return self.remove(frame)
        return cleaned_mask, frame_copy

    def check_adjust(self, output):
        """ Returns True if user deems cheekpad removal output as unacceptable. False otherwise."""
        self.adjust = False

        def left_right_click(event, x, y, flags, param):
            if event == 2:
                self.adjust = True
            if event == cv2.EVENT_LBUTTONDOWN:
                self.adjust = False

        window_name = 'is this cheekpad removal ok? left click if ok, right click to adjust; q to submit'

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, left_right_click)
        cv2.imshow(window_name, output)
        cv2.waitKey(0)

        return self.adjust


class HelmetSegment:
    """ HelmetSegment object that mediates finding and removing of helmet.
        One instance can be used across multiple subjects. (?)
    """

    def __init__(self):
        lower = [130, 120, 130]
        upper = [255, 255, 255]
        self.array_lower = np.array(lower, dtype="uint8")
        self.array_upper = np.array(upper, dtype="uint8")

    def helmet_tidy(self, helmet_segment):
        """ Returns a cleaned-up helmet_segment mask."""
        # closing the helmet segment to cover up as much edge as possible
        kernel = np.ones((12,12),np.uint8)
        helmet_segment_adj = np.array(helmet_segment, dtype=np.uint8)
        closed_segment = cv2.morphologyEx(helmet_segment_adj, cv2.MORPH_CLOSE, kernel)
        # make kernel smaller
        second_kernel = np.ones((8,8),np.uint8)
        final_segment = cv2.dilate(closed_segment,second_kernel,iterations = 1)
        final_segment = np.array(final_segment, dtype=bool)
        
        # clean up small contiguous regions of pixels that remain
        blobs, num_of_blobs = ndimage.label(final_segment)
        blob_areas = ndimage.sum(final_segment, blobs, index=range(blobs.max() + 1))
        # TODO scale to pixel density of frame
        blobs[np.where(blob_areas < 1000)] = 0 # removes any contiguous pixel regions with area < 200
        tidy_helmet = blobs > 0 # convert back to bool

        return tidy_helmet

    def find_helmet(self, image):
        """ Returns the helmet mask and masked image output."""

        color_mask = cv2.inRange(image, self.array_lower, self.array_upper)
        color_segment = color_mask > 0
        
        # define region that is redder than it is green
        red_green_segment = image[:,:,2] > image[:,:,1]
        
        # define the helmet as the part that's not very red, but is very bright
        mask = np.logical_and(np.invert(red_green_segment), color_segment)

        # masked image (no tidying done)
        output = cv2.bitwise_and(image, image, mask=mask.astype('uint8'))
        return mask, output

    def remove(self, image):
        """ Returns helmet mask and image with helmet removed.
            Mediates checking if output is acceptable.
        """
        mask, output = self.find_helmet(image)
        mask = self.helmet_tidy(mask)
        output = cv2.bitwise_and(image, image, mask=np.array(np.invert(mask), dtype='uint8'))
        if self.check_adjust(output):
            self.reset_params(image)
            self.avg_color = self.get_avg_color(frame)
            return self.remove(image)
        return mask, output

    def check_adjust(self, output):
        """ Returns True if user deems output of helmet removal as unacceptable. False if not."""

        self.adjust = False

        def left_right_click(event, x, y, flags, param):
            if event == 2:
                self.adjust = True
            if event == cv2.EVENT_LBUTTONDOWN:
                self.adjust = False

        window_name = 'is this helmet removal ok? left click if ok, right click to adjust; q to submit'

        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, left_right_click)
        cv2.imshow(window_name, output)
        cv2.waitKey(0)

        return self.adjust

    def reset_params(self, image):
        """ Reintialisation of self.array_lower and self.array_upper values by user manually."""

        tb_name = "adjust trackbar to mask helmet, press q when done"

        def callback(value):
            pass

        def setup_trackbars():
            """ Produces trackbars that can be used to adjust array_lower and array_upper values."""

            cv2.namedWindow(tb_name, 0)
            cv2.resizeWindow(tb_name, 800, 500)

            # only adjust lower color levels
            c = 0
            for j in 'BGR':
                v = self.array_lower[c]
                c += 1
                cv2.createTrackbar("{}_{}".format(j, 'MIN'), tb_name, v, 255, callback)

        def get_trackbar_values():
            """ Resets self.array_upper and/or self.array_lower on basis of trackbar position. """

            c = 0
            for j in "BGR":
                v = cv2.getTrackbarPos("{}_{}".format(j, "MIN"), tb_name)
                self.array_lower[c] = v
                c += 1
        
        mask, output = self.find_helmet(frame)
        setup_trackbars()  # the color sliders

        while True:
            get_trackbar_values()
            (mask, output) = self.find_helmet(frame)
            cv2.imshow(tb_name, output)  
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            
        cv2.destroyWindow(tb_name)
        cv2.destroyWindow("Preview")
        return (mask, output)
