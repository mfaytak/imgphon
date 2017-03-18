#!/usr/local/bin/python
# coding: utf-8

"""
Matthew Faytak 2-19-2016, mod. 2-8-2017
Usage: python batch-forced-aligner.py [subject directory]
Uses pyalign (based on HTK) through a subprocess to force-align an ultrasound acquisition.
Usage of pyalign assumes existence of dict.local file in the top-level directory, and a stim.txt 
file containing one of the dict.local file's key items in each directory.
Does not currently use ultratils but probably will be updated soon.
"""

import os, sys
import subprocess
import re

def usage():
    print("Usage: python batch-forced-aligner.py [subject directory]")

# French stimuli that happen to be in the CMU dictionary of Englsh with a different pronunciation.
# these are in the dict.local file with a -X at the end to make them totally unambiguous.
bad_stim = ["HAPPE","HABIT","ABU","ABOUT","ABBE","AVIS","BAVA"]
# a list of all French stimuli
# TODO replace functionality with runvars.phase attribute of acq objects
fr_stim = ["APIS","APU","PAPOU","HAPPE","APEU","APAUT","HABIT","ABU","ABOUT","ABBE","ABOEUFS","PASBEAU","AFFIT","AFFUT","PASFOU","MAFE","AFEU","PASFAUX","AVIS","AVU","AVOUS","AVEZ","AVEU","PAVOT","HAPPAIT","ABAIT","AFAIT","AVAIT","APAS","ABAS","AFA","BAVA"]

# throw error message if provided directory doesn't exist
try:
    directory = sys.argv[1]
except IndexError:
    usage()
    sys.exit(2)

# walk the directory structure and examine every filename
# TODO change to iterating over all acq in an experiment object
for root,dirs,files in os.walk(directory):  
    for soundfile in files:

        # if not a left-channel/channel 1 .wav or .WAV, go on to the next file
        if not 'ch1.wav' in soundfile.lower():  
            continue 

        # make a bunch of path string variables; locate stim file; notify user of what's up at that point
        wavFile = os.path.join(os.path.abspath(root),soundfile)
        parentDir = os.path.split(wavFile)[0]
        stimFile = os.path.join(parentDir,"stim.txt") 
        tsFile = os.path.join(parentDir,'ts.txt')

        try:
            test = open(stimFile,"r")
            print("Working with acquisition in " + stimFile)
        except IOError:
            print("WARNING: file {} not found".format(stimFile))

        with open (stimFile, "r") as myfile:
            key = myfile.read().rstrip('\n')
        if key == "bolus":
            continue
        if key in fr_stim: # different transcriptional fixes required for "French" words in Eng. HTK
            ts = key.upper().replace(" ","")
            if ts in bad_stim:
                ts = ts + "X" # e.g. HAPPEX, BAVAX
            with open(tsFile,"w") as out:
                out.write(ts)
            # name the transcription TextGrid and create it using the HTK force-aligner
            tg_out = os.path.join(os.path.abspath(root),str(os.path.splitext(soundfile)[0]+".TextGrid"))
            ret = subprocess.call(["pyalign", "-r", "16000", wavFile, tsFile, tg_out])
            print("Alignment in file {:}".format(tg_out))

        else:
            # name the transcription TextGrid and create it using the HTK force-aligner
            tg_out = os.path.join(os.path.abspath(root),str(os.path.splitext(soundfile)[0]+".TextGrid"))
            ret = subprocess.call(["pyalign", "-r", "16000", wavFile, stimFile, tg_out])
            print("Alignment in file {:}".format(tg_out))
