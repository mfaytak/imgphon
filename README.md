# imgphon
Python toolkit for articulatory **phon**etic analysis of **im**a**g**e data: lingual ultrasound and video of facial landmarks. Facial landmark detection is carried out using `dlib` on images preprocessed with `opencv3` and `ndimage`. (Ultrasound feature extraction TBA.) Dimensionality reduction is carried out using `scikit-learn`.

Files in the `scripts` directory also carry out forced alignment and acoustic analysis routines from various stages of writing my dissertation; these and the ultrasound analysis routines they coincide with will eventually be converted to functions in a sensible namespace.

# Getting started
### Installing dependencies
The following dependencies, among others, are required for facial feature extraction using the functions and classes in `imgphon/imgphon`: `opencv3`, `imutils`, and `dlib`. The dependencies for `dlib` are especially large and complex. We have used the installation instructions for dlib on Unix-type systems at bit.ly/2HEhCQP with some success. The instructions provided at `davisking/dlib` may also help.

### Configuration
To configure upon downloading the package, simply navigate to the root directory of the repo and run `python setup.py install`.

### Demos
A demonstration of lip shape extraction and dimensionality reduction can be found in **lip-demo.ipynb**. A demonstration of some tools for extracting and processing ultrasound video frames from the Telemed EchoBlaster's binary outputs can be found in **echob-data-demo.ipynb**. Use of these extracted features for analysis coming soon. 

### Usage
*Coming soon*

