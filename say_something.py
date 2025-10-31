import wave

from piper import PiperVoice
from piper import SynthesisConfig

voice = PiperVoice.load("my-model/en_US-W7IY.onnx")

syn_config = SynthesisConfig(
    volume=0.85,  # half as loud
    length_scale=1,  # twice as slow
    noise_scale=0.667,  # more audio variation
    noise_w_scale=0.9,  # more speaking variation
    normalize_audio=True, # use raw audio from voice
)


with wave.open("test.wav", "wb") as wav_file:
    voice.synthesize_wav("Juliet got an x-ray of her broken leg!", wav_file,syn_config=syn_config)
