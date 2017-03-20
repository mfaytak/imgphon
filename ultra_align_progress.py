'''
Last modified 3-20-2017, Matt Faytak.
Praat script and portions of Python are from Jevon Heath, Stephen Ho, Michelle Ching.
This script is a utility for processing TextGrid files in an ultrasound directory structure via Praat;
it generates and stores in plaintext files a subject-specific list of audio files, as well as where 
you left off, if you choose to stop execution of the script and come back later.
'''

import os, sys, glob

def usage():
    print("Usage: python align_progress.py [subject directory]")

try:
	expdir = sys.argv[1]
except IndexError:
	usage()
	sys.exit(2)

try:
	os.chdir(expdir)
except OSError:
	usage()
	sys.exit(2)

# generates list of ch1.wav files.
wav_list = "./listofwavs.txt"
with open(wav_list, "w+") as f:  
	for w in glob.glob("./*/*/*ch1.wav"):
		try:
			tGrid = w.replace('.wav', '.TextGrid')
			if os.path.isfile(tGrid):
				file_str = w.replace('./','')
				f.write(file_str + "\n")
		except OSError:
			print('Cannot open WAV file ', w)

# generates start variable as "bookmark" of progress
start_file = "./start.txt"
with open(start_file, "w+") as n:
	n.write("1")

# generates Praat script.
# once this has run, open Praat and execute the Praat script.
praat_script = "./rapidEditSp17.praat"
with open(praat_script, "w+") as f:
	f.write("# This script opens all sound files in the specified text file and associated TextGrids,\n")
	f.write("# pauses for the user to check and possibly alter the TextGrid,\n")
	f.write("# then saves the new TextGrid in the same directory if applicable.\n\n")
	f.write("# Jevon Heath, 5/6/15\n")
	f.write("#Script adapted by Steven Ho and Michelle Ching\n")
	f.write("# Create a list of all sound files in the directory.\n\n")
	f.write("form Check Textgrids for .wav files listed in this file\n")
	f.write("\tcomment Name of directory:\n")
	wd = os.path.dirname(os.path.realpath(__file__))
	f.write("\ttext directory " + wd + "\n")
	f.write("\tcomment Name of text file:\n")
	f.write("\ttext textfile listofwavs.txt\n")
	f.write("endform\n\n")
	f.write("Read Strings from raw text file... 'directory$'/'textfile$'\n")
	f.write("fileList$ = selected$ (\"Strings\")\n")
	f.write("n = Get number of strings\n")
	f.write("history$ = \"start\"\n")
	f.write("start = readFile (history$)\n\n")
	f.write("# Open each sound file and its associated TextGrid so the TextGrid can be manipulated.\n")
	f.write("# Once the user clicks \"Continue\", the TextGrid is saved and the next soundfile is opened.\n\n")
	f.write("for i from start to n\n")
	f.write("\tselect Strings 'fileList$'\n")
	f.write("\tsoundfile$ = Get string... i\n")
	f.write("\tRead from file... 'directory$'/'soundfile$'\n")
	f.write("\tsoundname$ = selected$ (\"Sound\")\n")
	f.write("\tacqnumber$ = left$ (soundname$, 22)\n")
	f.write("\tunderscoretgname$ = soundname$\n")
	f.write("\tdottgname$ = acqnumber$ + \".bpr.ch1\"\n")
	# f.write("\tdottgname$ = replace$ (underscoretgname$, \"_bpr_ch1\", \".bpr.ch1\", 1)\n")
	f.write("\tRead from file... 'directory$'/'acqnumber$'/'dottgname$'.TextGrid\n")
	f.write("\tplus Sound 'soundname$'\n")
	f.write("\tdo(\"View & Edit\")\n")
	f.write("\tbeginPause(\"Any edits?\")\n")
	f.write("\tclicked = endPause(\"Continue\", \"Save and Continue\", 1)\n")
	f.write("\tselect TextGrid 'underscoretgname$'\n")
	f.write("\tif clicked = 2\n")
	f.write("\t\tSave as text file... 'directory$'/'acqnumber$'/'underscoretgname$'.TextGrid\n")
	f.write("\tendif\n")
	f.write("\tplus Sound 'soundname$'\n")
	f.write("\tRemove\n")
	f.write("\tdeleteFile: history$\n")
	f.write("\twriteFile: history$, i+1\n")
	f.write("endfor")
