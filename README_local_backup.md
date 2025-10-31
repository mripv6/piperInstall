# Clone the installation repo into your computer

# Unpack tar file
cd ~
mkdir temp
cp piper-w7iy.tar.gz temp
cd temp
tar xvf piper-w7iy.tar.gz

#Run setup script
Run the install script. This script sets up the environment for recording,
training, exporting and testing the piper voice model. This uses the GPL
version of piper and python 3.13. The resulting voice model will be compatible
with the MIT licensed version of piper used in N1MM.

install.sh

# Convert the epoch file
This utility was built by PE1EEC. It converts the old ryan checkpoint file
to a newer format. This utility may go away as piper gpl improves. Or different
checkpoint files are created

python checkpoint_convert.py

# record the wav files
Use this utility to record wav files used for training. The wav files are
placed in my-training/wav so you don't accidentally write over your existing
training dataset. Once you're happy with the wav files, copy everying from the
my-training/wav directory to the dataset directory.

python record.py
cp ~/piper1-gpl/my-training/wav/* ~/piper1-gpl/dataset/

# train the model
Finally! The following command trains the voice model. You can tweak the 
values in the training.yaml file, if needed. These values worked for me based
on info from Claude and the resources available in my system. The existing
config file will training over 500 epochs, which takes my computer about 45 minutes.

python -m piper.train fit --config training.yaml

# test the model
Use the followig utility to export the voice model to onnx format, gather the
config json file, rename the voice model files and develop a test wav file.

python export_and_test.py --name w7iy \
  --text "CQ Contest! Whiskey 4 november fox, whiskey four november foxtrot!"

aplay ./my-model/test_w7iy.wav

After you run this script, you can create test wav files by modifying the 
say_something.py script. Then run. This is a good example of using the voice
model in a python script.

python say_something.py

# Copy voice model to PC with N1MM
Copy the onnx and json file to the piperModel subdirectory on your PC. This voice
model should show up in the N1MM configuration after a N1MM restart. I use a USB stick.

# Files included:

README.md              ; This file
piper-stu.txt          ; Extensive notes on this project. Use for manual install.
install.sh             ; Installation script
checkpoint_convert.py  ; Convert old voice checkpoint files, may not need in future
training.yaml          ; Training configuration file, edit as needed before training
recording.py           ; Script to help record training wav files
sentences.txt          ; Sentences used by recording script. Modify as necessary
export_and_test.py     ; Export to onnx model and gather json file, then create a test wav file
audio_logger_callback.py ; Experimental, used to send audio to tensor board
debug_callback.py      ; Experimental. Used to debug callback functions during training
say_something.py       ; Used to create a test wav file. Use after export.


