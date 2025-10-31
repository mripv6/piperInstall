Set up Piper Training Environment for N1MM
==========================================

This script sets up the environment for recording, training, exporting 
and testing the piper voice model. I decided to use python 3.13 and the GPL
version of piper. The resulting voice model will be compatible
with the MIT licensed version of piper used in N1MM. These scripts have
been tested under Linux Mint 22. They may work under other versions
of linux and perhaps Windows WSL.

Training a model using 45 sentences takes about 45 minutes on my Linux computer, 
which has a RTX4060TI GPU and 16 thread CPU. I decided to use more sentences
than the standard 7 used by K3CT.

Consult the file piper-stu.txt for detailed information. That file was started
using the instuctions from K3CT. I modifed and added to the instructions as
I learned more. I also created several scripts to help automate tasks. Hopefully,
this will be as easy as installing the environment, recording wav files, training
and testing. Once satisfied, copy the voice model files to the PC where N1MM+ runs.

# Getting Started

Clone this repository into your home directory. Then run the installation
shell script.

```
install.sh
```

# Convert the epoch file
This utility was built by PE1EEC. It converts the old ryan checkpoint file
to a newer format. This utility may go away as piper gpl improves or different
checkpoint files are created

```
python checkpoint_convert.py
```

# Record the wav files
Use this utility to record wav files used for training. The wav files are
placed in my-training/wav so you don't accidentally write over your existing
training dataset. Once you're happy with the wav files, clear out the exisiting
data set and copy everying from the my-training/wav directory to the dataset 
directory. This utility also creates the metadata.csv file required by piper.

```
python record.py
cp ~/piper1-gpl/my-training/wav/* ~/piper1-gpl/dataset/
```

# train the model
Finally! The following command trains the voice model. You can tweak the 
values in the training.yaml file, if needed. These values worked for me based
on info from Claude and the resources available in my system. The existing
config file will training over 500 epochs, which takes my computer about 45 minutes.

```
python -m piper.train fit --config training.yaml
```

# test the model
Use the followig utility to export the voice model to onnx format, gather the
config json file, rename the voice model files and develop a test wav file.

```
python export_and_test.py --name w7iy \
  --text "CQ Contest! Whiskey 4 november fox, whiskey four november foxtrot!"

aplay ./my-model/test_w7iy.wav
```

After you run this script, you can create test wav files by modifying the 
say_something.py script. Then run. This is a good example of using the voice
model in a python script.

```
python say_something.py
```

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


