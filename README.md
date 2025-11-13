Set up Piper Training Environment for N1MM
==========================================

# Introduction

This project sets up the environment for recording, training, exporting 
and testing the piper voice model. I decided to use python 3.13 and the GPL
version of piper. The resulting voice model will be compatible
with the MIT licensed version of piper used in N1MM+. These scripts have
been tested under Linux Mint 22 and Windows WSL Ubuntu 24. (See note below.) 

Others are working on scripts to use with
Windows 11, which probably makes more sense because N1MM also runs under Windows and
users aren't likely to have multiple computers. My linux computer has more horse power 
and I have some other AI based projects in mind. 

Consult the file piper-stu.txt for detailed information. K3CT's instructions served
as a starting point. Claude AI was a big help, but I still had to troubleshoot. It 
would have taken weeks for me to write them from scratch. Thanks to PE1EEC 
for his help and checkpoint conversion script.

If you get stuck, consult my extensive notes in docs/piper-stu.txt. There are little
tidbits in there that may help.

A note about running under WSL - The installation script, checkpoint conversion script
and the training command all work under WSL. The WAV recording utility does NOT without
a lot of work. So you may want to use K3CT's recording script or Audacity. If you do, the 
wav directories are different. See the note in the recording section below.

The basic steps are:

- Clone the Repo
- Install the environment
- Convert the check point file
- Record the wav files
- Train the model
- Test

# Clone this Repo

Use git to clone the repo. This will create a subdirectory called ~/piperInstall with all
the files you need.

```
cd ~
git clone https://github.com/mripv6/piperInstall.git
```

# Install the environment

Run the installation shell script.

```
cd ~/piperInstall
./install.sh
```
Running this file will take a long time. You will need to enter your sudo 
password because it's installing several packages. After that no user interaction
is requied. The script will also set up
the right subdirectories, clone the repo, download the checkpoint files and setup
the python environment.

# Convert the checkpoint file
This utility was built by PE1EEC. It converts the old ryan checkpoint file
to a newer format. This utility may go away as piper gpl improves or different
checkpoint files are created. The script will bring up a GUI. You will have two
tasks. First, the app asks for the file to convert. Navigate to the 
checkpoint file under lightning_logs/version_0/checkpoints. Select the checkpoint 
file. Second, set the target directory to /lightning_logs/version_0/checkpoints.
Note, you have to change directory to piper1-gpl and activate the python
virtual environment. (You'll see (.venv) before your linux prompt.

```
cd ~/piper1-gpl
source ~/piper1-gpl/src/python/.venv/bin/activate"
python checkpoint_convert.py
```

# Record the wav files
Use this utility to record wav files used for training. The wav files from this
utility  are
placed in my-training/wav so you don't accidentally write over your existing
training dataset. Once you're happy with the wav files, copy everying from the 
my-training/wav directory to the dataset 
directory. This utility creates the metadata.csv file required by piper, which
goes in the same directory as the wav files. record.py does NOT work under WSL.

The utility uses TK for the GUI. The user interface is simple and should be
intuitive. You can delete wav files during the review. Run the script again
to re-record missing files.

```
cd ~/piper1-gpl
source ~/piper1-gpl/src/python/.venv/bin/activate"
python record.py
```
Once you're happy with all the WAV file recordings, copy them to the dataset
subdirectory.  
```
cp ~/piper1-gpl/my-training/wav/* ~/piper1-gpl/dataset/
```
I used an external sound card made by behringer and a heil desktop microphone. 
I chose the sound card to connect to the microphone in Sound Preferences. The levels
were set pretty high, but I avoided clipping. The utility trims silence from
the beginning and end. It also normalizes the wav file so all files have the
same volume level.

Note 1 - If you're running this under WSL, my record.py script won't work. You have
to install pulse audio, which I haven't tested. So use K3CT's recordinig script
instead. When you finish producing the wav files, they go under piper1-gpl/dataset/wav.
Save metadata.csv to piper1-gpl/dataset. Check the metadata.csv file for the directory
paths, if something breaks. You can also use Audacity, but need to build metadata.csv
by hand. Hint, the Windows C drive is under /mnt/c.  

Note 2 - I'm still experimenting with training sentences. K3CT uses 7, but my research
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

Note, you need to run the training from the piper1-gpl directory with the 
python virtual environment activated.

```
cd ~/piper1-gpl
source ~/piper1-gpl/src/python/.venv/bin/activate"
```
Edit training.yml to match your environment. Check the accelerator (cpu or cuda) and
file paths to the processed checkpoint. If the training process throws an error, you 
may have a problem with the path.

```
nano training.yaml
python -m piper.train fit --config training.yaml
```

There are some experimental callbacks that can be used in order to expose the trained audio
file to Tensorboard. But I'm not sure they are working correctly. Until they are
provent to work, you can still export and test during training using the export
script. (See my detailed notes for setting up Tensorboard. It wasn't too helpful.)

# Test the model
Use the followig utility to export the voice model to onnx format, gather the
config json file, rename the voice model files and develop a test wav file. ONNX
is a standard format for voice models and stands fro Open Neural Network Exchange. There
are other command line switches you can play with.

Note - export_and_test.py will work under WSL. However, aplay will not. So you have
to copy the test wav file somewhere under Windows to try.

```
cd ~/piper1-gpl
source ~/piper1-gpl/src/python/.venv/bin/activate"
python export_and_test.py --name w7iy \
  --text "CQ Contest! Whiskey 4 november fox, whiskey four november foxtrot!"

aplay ./my-model/test_w7iy.wav
```

After you run this script, you can create test wav files by modifying the 
say_something.py script. Then run. This is a good example of using the voice
model in a python script. The --name parameter comes from voice_name parameter
in training.yaml. (This doesn't work under WSL.)

```
python say_something.py
```

# Copy voice model to PC with N1MM
Copy the onnx and json files in the my-model directory to the piperModel 
subdirectory on your PC. This voice model should show up in the N1MM configuration 
after a N1MM restart. I use a USB stick. If you're using WSL, hint: /mnt/c is the 
Windows C: drive.

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


