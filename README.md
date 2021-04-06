## test from AIVE
Draw a box around each person in videos. Just put all videos in the "input" directory, run the script 
and the output videos will be in the "output" directory

Some parameters can be changed, in particular :
PERIOD: to process the video faster, not all frames are tested. Only 1 frame every [PERIOD] 
will be used to draw the boxes and the boxes will be caried over the other frames.

MIN_SCORE: The minimal score to detect a person. A higher MIN_SCORE will reduce the number of false positive

#### Installation
There is a requirements.txt but "tensorflow" added a bunch of libraries, 
the only libraries really installed are:  
opencv  
tensorflow  
tensorflow_hub  