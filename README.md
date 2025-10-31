Set up Piper Training Environment for N1MM
==========================================

# Introduction

This script sets up the environment for recording, training, exporting 
and testing the piper voice model. I decided to use python 3.13 and the GPL
version of piper. The resulting voice model will be compatible
with the MIT licensed version of piper used in N1MM. These scripts have
been tested under Linux Mint 22. They may work under other versions
of linux and perhaps Windows WSL. 

Others are working on scripts to use with
Windows 11, which probably makes more sense because N1MM also runs under Windows and
users aren't likely to have multiple computers. I used my
Linux computer because it has more horse power. Plus, I have some other AI based
projects in mind. I considered using Docker, but decided to avoid the extra work.

Training a model using 45-50 sentences takes about 45 minutes on my Linux computer, 
which has a RTX 4060TI GPU and 16 thread CPU. I decided to use more sentences
than the standard 7 used by K3CT. This is a work in progress. 

Consult the file piper-stu.txt for detailed information. That file was started
using the instuctions from K3CT. I modifed and added to the instructions as
I learned more. I also created several scripts to help automate tasks. Hopefully,
this will be as easy as installing the environment, recording wav files, training
and testing. Once satisfied, copy the voice model files to the PC where N1MM+ runs.

I need to thank Claude AI and ChatGPT for assistance! Claude was awesome for developing
the scripts. I still had to troubleshoot, but it would have taken weeks for me
to write them from scratch. Be careful with ChatGPT, it only remembers so far back into the chat.
It will rewrite parts of the script that worked. Also, thanks to K3CT and PE1EEC for
their help.

If you get stuck, consult my extensive notes in docs/piper-stu.txt. There are little
tidbits in there that may help.

# Clone this Repo

Use git to clone the repo. This will create a subdirectory called piperInstall with all
the files you need.

```
cd ~
git clone https://github.com/mripv6/piperInstall.git
```

# Install the environment

Run the installation shell script.

```
cd ~/piperInstall
install.sh
```
Running this file will take some time. It will require your sudo username and 
password because it's installing several packages. After that no user interaction
is requied. The script will also set up
the right subdirectories, clone the repo, download the checkpoint files and setup
the python environment.

# Convert the epoch file
This utility was built by PE1EEC. It converts the old ryan checkpoint file
to a newer format. This utility may go away as piper gpl improves or different
checkpoint files are created. The script will bring up a GUI. Navigate to the
checkpoint file under lightning_logs, version_0 and checkpoints. Select the 
file. Then navigate back to the same directory to store the processed file.

```
python checkpoint_convert.py
```

# Record the wav files
Use this utility to record wav files used for training. The wav files are
placed in my-training/wav so you don't accidentally write over your existing
training dataset. Once you're happy with the wav files, clear out the exisiting
data set and copy everying from the my-training/wav directory to the dataset 
directory. This utility also creates the metadata.csv file required by piper.

The utility uses TK for the GUI. The user interface is simple and should be
intuitive. You can delete wav files during the review. Run the script again
to re-record missing files.

```
python record.py
cp ~/piper1-gpl/my-training/wav/* ~/piper1-gpl/dataset/
```

I used an external sound card made by behringer and a heil desktop microphone. 
I chose the sound card to connect to the microphone in Sound Preferences. The levels
were set pretty high, but I avoided clipping. The utility trims silence from
the beginning and end. It also normalizes the wav file so all files have the
same volume level.

Note - I'm still experimenting with training sentences. K3CT uses 7, but my research
shows I should use many more. You can change the sentences in 'sentences.txt'.
Whatever you do, make sure you emphasis words related to those you want to use in
N1MM. For example, if you have a sentence like CQ Contest - say it like you do
during a contest. That way the inflection in the text to speech file will turn
out great.

# Train the model
Finally! The following command trains the voice model. You can tweak the 
values in the training.yaml file, if needed. These values worked for me based
on info from Claude and the resources available in my system. The existing
config file will training over 500 epochs, which takes my computer about 45 minutes.

```
python -m piper.train fit --config training.yaml
```

If you get any errors, you may have missing files or directories. There are some
experimental callbacks that can be used in order to expose the trained audio
file to Tensorboard. But I'm not sure they are working correctly. Until they are
provent to work, you can still export and test during training using the export
script.

# Test the model
Use the followig utility to export the voice model to onnx format, gather the
config json file, rename the voice model files and develop a test wav file. ONNX
is a standard format for voice models and stands fro Open Neural Network Exchange.

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
Copy the onnx and json files in the my-model directory to the piperModel 
subdirectory on your PC. This voice model should show up in the N1MM configuration 
after a N1MM restart. I use a USB stick.

# Files included:

- README.md              ; This file
- piper-stu.txt          ; Extensive notes on this project. Use for manual install.
- install.sh             ; Installation script
- checkpoint_convert.py  ; Convert old voice checkpoint files, may not need in future
- training.yaml          ; Training configuration file, edit as needed before training
- recording.py           ; Script to help record training wav files
- sentences.txt          ; Sentences used by recording script. Modify as necessary
- export_and_test.py     ; Export to onnx model and gather json file, then create a test wav file
- audio_logger_callback.py ; Experimental, used to send audio to tensor board
- debug_callback.py      ; Experimental. Used to debug callback functions during training
- say_something.py       ; Used to create a test wav file. Use after export.


