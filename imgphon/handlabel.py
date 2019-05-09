import os
import sys
import re
import contextlib

import numpy as np
import cv2
import easygui

import imgphon


def interface(windowName, canvas, tp, frame_index, file_name, result_dict, clicks_list, working_dir, sort_fn, label_num_constr: int, ext):
    output_imgs_dir = "output_imgs"

    with contextlib.suppress(FileExistsError):
        os.mkdir(os.path.join(working_dir, output_imgs_dir))

    shouldReload = False
    output_img_name = str(file_name) + "-" + str(frame_index) + "-" + str(tp)
    while True:
        cv2.imshow(windowName, canvas)
        key = cv2.waitKey(1) & 0xFF  # ASCII code of pressed key
        if key == 110:  # [N] to next
            if label_num_constr and (len(clicks_list) != label_num_constr):
                shouldReload = True
                break
            cv2.imwrite(os.path.join(working_dir, output_imgs_dir,
                                     output_img_name + ext), canvas)
            result_dict[tp] = sort_fn(clicks_list)
            break
        elif key == 114:  # [R] to reload
            shouldReload = True
            break
        elif key == 113:  # [Q] to quit
            if label_num_constr and (len(clicks_list) == label_num_constr):
                cv2.imwrite(os.path.join(working_dir,
                                         output_imgs_dir, output_img_name + ext), canvas)
                result_dict[tp] = sort_fn(clicks_list)
            cv2.destroyAllWindows()
            np.save(os.path.join(working_dir,
                                 "result_dict.npy"), result_dict, allow_pickle=True)
            os.remove("temp.bmp")
            sys.exit(0)
    cv2.destroyAllWindows()
    return (shouldReload, result_dict)


def paint_dot(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param[0], (x, y), 3, (60, 20, 220), -1)
        param[1].append((x, y))


def label_single(frame_index: int, vid_path: str, total_frames_count: int, result_dict: dict, working_dir: str, sort_fn, label_num_constr: int, ext, tp_list):
    shouldReload = True
    tp = tp_list[frame_index - 1]
    file_name = os.path.basename(vid_path)
    imgphon.get_video_frame(vid_path, tp)
    m_clicks_list = []

    while shouldReload:
        curr_frame = cv2.imread("temp.bmp")
        windowName = "(Frame " + str(frame_index) + " of " + str(total_frames_count) + ") " + str(file_name) + " @ " + str(tp) +  \
            " Sec | [R] Reload | [N] Save & Next | [Q] Save & Quit"
        cv2.namedWindow(windowName)
        cv2.moveWindow(windowName, 320, 180)
        cv2.setMouseCallback(windowName, paint_dot,
                             (curr_frame, m_clicks_list))

        interface_return = interface(windowName, curr_frame,
                                     tp, frame_index, file_name, result_dict, m_clicks_list, working_dir, sort_fn, label_num_constr, ext)
        shouldReload = interface_return[0]
        if shouldReload:
            m_clicks_list = []

    os.remove("temp.bmp")
    return interface_return[1]


def label_multiple(start_pt: int, tp_list: list, vid_path: str, result_dict: dict, working_dir: str, sort_fn, label_num_constr: int, ext):
    total_frames_count = str(len(tp_list))
    tp_index = start_pt
    while tp_index < len(tp_list):
        m_tp_index = tp_index + 1
        new_result_dict = label_single(
            m_tp_index, vid_path, total_frames_count, result_dict, working_dir, sort_fn, label_num_constr, ext, tp_list)
        result_dict = new_result_dict
        tp_index += 1


def sort_lip_coords(tmp_l):
    def take_x(elem):
        return elem[0]
    tmp_l.sort(key=take_x)

    left = tmp_l[0]
    right = tmp_l[3]
    if tmp_l[1][1] < tmp_l[2][1]:
        upper = tmp_l[1]
        lower = tmp_l[2]
    else:
        upper = tmp_l[2]
        lower = tmp_l[1]

    tmp_dict = {"leftx": left[0],
                "lefty": left[1],
                "rightx": right[0],
                "righty": right[1],
                "upperx": upper[0],
                "uppery": upper[1],
                "lowerx": lower[0],
                "lowery": lower[1]}

    return tmp_dict
