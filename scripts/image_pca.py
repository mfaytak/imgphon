"""
5-9-2016 updated 3-9-2017 Matthew Faytak
This script reads midpoint frames in from an experiment, performs PCA on the image data, 
and returns a .csv file containing PC loadings and associated metadata. Currently only runs on one subject at a time.
This script contains the basic functionality of ultra_imagepca.py, but less tailored to my experiments' data
(e.g. does not subset data in the way I need).
----
Expected usage if stored in processing dir: $ python image_pca.py (-f -v -r -c) directory num_components
----
"""
from __future__ import absolute_import, division, print_function

import os
import re
import numpy as np
from ultratils.exp import Exp
import audiolabel
import argparse

# for plotting
import matplotlib.pyplot as plt

# for PCA business
from sklearn import decomposition
from sklearn.decomposition import PCA

# regular expression for target segments
vre = re.compile(
         "^(?P<vowel>AA|AE|AH|AO|EH|ER|EY|IH|IY|OW|UH|UW)(?P<stress>\d)?$"
      )

# Read in and parse the arguments, getting directory info and whether or not data should flop
parser = argparse.ArgumentParser()
parser.add_argument("directory", help="Experiment directory containing all subjects")
parser.add_argument("num_components", type=int, help="Number of principal components to output")
parser.add_argument("-v", "--visualize", help="Produce plots of PC loadings on fan",action="store_true")
parser.add_argument("-f", "--flop", help="Horizontally flip the data", action="store_true")
parser.add_argument("-c", "--convert", help="Scan-convert the data before analysis", action="store_true")
parser.add_argument("-r", "--no_er", help="Run PCA without schwar in data set.", action="store_true")
args = parser.parse_args()

try:
    expdir = args.directory
except IndexError:
    print("\tDirectory provided doesn't exist!")
    ArgumentParser.print_usage
    ArgumentParser.print_help
    sys.exit(2)

e = Exp(expdir=args.directory)
e.gather()

# check for appropriate number of components
if args.num_components > (len(e.acquisitions) - 1):
    print("EXITING: Number of components requested definitely exceeds number to be produced")
    sys.exit(2)
    
# create output file path
pc_out = os.path.join(e.expdir,"pc_out.txt")

frames = None
threshhold = 0.020 # threshhold value in s for moving away from acoustic midpoint measure

# subject = [] TODO add if subject loop is added 
phase = []
trial = []
phone = []
tstamp = []

if args.convert:
    conv_frame = e.acquisitions[0].image_reader.get_frame(0)
    conv_img = test.image_converter.as_bmp(conv_frame)

for idx,a in enumerate(e.acquisitions):

    if frames is None:
        if args.convert:
            frames = np.empty([len(e.acquisitions)] + list(conv_img.shape))
        else:
            frames = np.empty([len(e.acquisitions)] + list(a.image_reader.get_frame(0).shape)) * np.nan

    tg = str(a.abs_image_file + ".ch1.TextGrid")
    pm = audiolabel.LabelManager(from_file=tg, from_type="praat")
    v,m = pm.tier('phone').search(vre, return_match=True)[-1] # return last V = target V
    word = pm.tier('word').label_at(v.center).text
    
    phase.append(a.runvars.phase)
    trial.append(idx)
    tstamp.append(a.timestamp)

    # HOOF fix - CMUdict has UW1 for the word
    if word == "HOOF":
        phone.append("UH1")
    else:
        phone.append(v.text)
    
    if args.convert:
        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev", convert=True)
    else:
        mid, mid_lab, mid_repl = a.frame_at(v.center,missing_val="prev")

    if mid is None:
        if mid_repl is None:
            print("SKIPPING: No frames to re-select in {:}".format(a.timestamp))
            continue
        else:
            if abs(mid_lab.center - v.center) > threshhold:
                print("SKIPPING: Replacement frame past threshhold in {:}".format(a.timestamp))
                continue
            else:
                mid = mid_repl
                
    frames[idx,:,:] = mid

# # # generate PCA objects # # #

# an example of subsetting.
# remove all schwars, if desired (all ER1 outside of learning phase)
if args.no_er:
    isnt_er = [f != "ER1" for f in phone]
    is_learning = [p == "learning" for p in phase]
    isnt_schwar = [a or b for a,b in zip(isnt_er,is_learning)]
    # reduce size of array that PCA is to be run on
    frames = np.squeeze(frames.take(np.where(isnt_schwar),axis=0))
    phase = np.squeeze(np.array(phase).take(np.where(isnt_schwar),axis=0))
    trial = np.squeeze(np.array(trial).take(np.where(isnt_schwar),axis=0))
    phone = np.squeeze(np.array(phone).take(np.where(isnt_schwar),axis=0))
    tstamp = np.squeeze(np.array(tstamp).take(np.where(isnt_schwar),axis=0))

# remove any indices for all objects generated above where frames have NaN values (due to skipping or otherwise)
keep_indices = np.where(~np.isnan(frames).any(axis=(1,2)))[0]
kept_phone = np.array(phone,str)[keep_indices] 
kept_trial = np.array(trial,str)[keep_indices]
kept_phase = np.array(phase,str)[keep_indices]
kept_frames = frames[keep_indices]
kept_tstamp = tstamp[keep_indices]

n_components = args.num_components
pca = PCA(n_components=n_components)
frames_reshaped = kept_frames.reshape([kept_frames.shape[0], kept_frames.shape[1]*kept_frames.shape[2]])

pca.fit(frames_reshaped)
analysis = pca.transform(frames_reshaped)

meta_headers = ["phase","trial","timestamp","phone"]
pc_headers = ["pc"+str(i+1) for i in range(0,n_components)] # determine number of PC columns; changes w.r.t. n_components
headers = meta_headers + pc_headers

out_file = args.outfile

d = np.row_stack((headers,np.column_stack((kept_phase,kept_trial,kept_tstamp,kept_phone,analysis))))
np.savetxt(out_file, d, fmt="%s", delimiter =',')

print("Data saved. Explained variance ratio of PCs: %s" % str(pca.explained_variance_ratio_))

# # # output images describing component min/max loadings # # #

if args.visualize:
    image_shape = (416,69)

    for n in range(0,n_components):
        d = pca.components_[n].reshape(image_shape)
        mag = np.max(d) - np.min(d)
        d = (d-np.min(d))/mag*255
        pcn = np.flipud(e.acquisitions[0].image_converter.as_bmp(d)) # converter from any frame will work; here we use the first

        if args.flop:
            pcn = np.fliplr(pcn)

        plt.title("PC{:} min/max loadings".format(n+1))
        plt.imshow(pcn, cmap="Greys_r") 
        savepath = "subj5-pc{:}.pdf".format(n+1)
        plt.savefig(savepath)

