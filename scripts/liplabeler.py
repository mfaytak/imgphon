import os
import sys
import re
import contextlib

import numpy as np
import easygui

from handlabel import *

if __name__ == "__main__":

    m_tp_list = [5.000, 10.110]

    result_dict = {}

    all_mode = "Start a new annotation"
    cont_mode = "Resume an unfinished work"
    single_mode = "Modify a single frame"

    welcome_msg = """Please select an option to proceed.\n
    NOTICE: You may only modify a single frame for annotations
    that you already started or finished."""

    choice = easygui.buttonbox(
        msg=welcome_msg, title="LipLabeler", choices=(all_mode, cont_mode, single_mode))
    if choice == None:
        sys.exit(0)

    vid_path = easygui.fileopenbox(title="Select a video file (*.MOV)...")
    if vid_path == None:
        sys.exit(0)

    working_dir = re.sub(r'\..*', '', vid_path)
    with contextlib.suppress(FileExistsError):
        os.mkdir(working_dir)

    def output_quality():
        ext = easygui.choicebox(msg="Select desired output image quality: \n \
        .bmp: large file, high quality \n \
        .png: smaller file, high quality \n \
        .jpg: smallest file, low quality",
                                title="Select Quality", choices=[".bmp", ".png", ".jpg"], preselect=1)
        if ext == None:
            sys.exit(0)
        else:
            return ext

    # ALL MODE
    if choice == all_mode:
        ext = output_quality()
        label_multiple(0, m_tp_list, vid_path, result_dict,
                       working_dir, sort_lip_coords, 4, ext)
    # RESUME MODE
    elif choice == cont_mode:
        with contextlib.suppress(FileNotFoundError):
            result_dict = np.load(os.path.join(
                working_dir, "result_dict.npy"), allow_pickle=True).item()
        start_tp = 0
        tp_index = 0
        while tp_index < len(m_tp_list):
            tp = m_tp_list[tp_index]
            if not tp in result_dict:
                start_tp = tp_index
                break
            tp_index += 1
        ext = output_quality()
        label_multiple(start_tp, m_tp_list, vid_path,
                       result_dict, working_dir, sort_lip_coords, 4, ext)
    # SINGLE MODE
    elif choice == single_mode:
        try:
            result_dict = np.load(os.path.join(
                working_dir, "result_dict.npy"), allow_pickle=True).item()
        except FileNotFoundError:
            if easygui.msgbox(msg="""Single mode is only for modifying existing project. \
                                     You need to start one first.""") == "OK":
                sys.exit(0)
        frame_index = easygui.integerbox(
            msg="Please enter the frame index you would like to modify", title='Frame Timepoint',
            lowerbound=1, upperbound=(len(m_tp_list)))
        if frame_index == None:
            sys.exit(0)
        ext = output_quality()
        label_single(frame_index, vid_path, 0, result_dict,
                     working_dir, sort_lip_coords, 4, ext, m_tp_list)

    # WRITING TO OUTPUT
    keys = ["leftx", "lefty", "rightx", "righty",
            "upperx", "uppery", "lowerx", "lowery"]
    open(os.path.join(working_dir, "result.txt"), "w+").close()
    with open(os.path.join(working_dir, "result.txt"), "a+") as f:
        f.write("index\ttimestamp\t")
        for key in keys:
            f.write(key + "\t")
        f.write("\n")
        tp_index = 0
        while tp_index < len(m_tp_list):
            tp = m_tp_list[tp_index]
            f.write(str(tp_index + 1) + "\t")
            f.write(str(tp))
            for key in keys:
                f.write("\t")
                f.write(str(result_dict[tp][key]))
            f.write("\n")
            tp_index += 1

    np.save(os.path.join(working_dir,
                         "result_dict.npy"), result_dict, allow_pickle=True)
