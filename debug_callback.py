from lightning.pytorch.callbacks import Callback

class DebugCallback(Callback):
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step == 20:  # Only debug at step 20
            print("\n=== DEBUG INFO ===")
            print(f"Batch type: {type(batch)}")
            print(f"Batch attributes: {dir(batch)}")
            
            # Check common attribute names
            for attr in ['audio', 'wav', 'waveform', 'y', 'mel', 'text', 'phonemes', 
                        'audio_norm', 'spec', 'spectrogram', 'mels']:
                if hasattr(batch, attr):
                    value = getattr(batch, attr)
                    if hasattr(value, 'shape'):
                        print(f"  batch.{attr}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  batch.{attr}: type={type(value)}")
            
            # Try to get the actual audio from the model
            print("\n--- Trying to generate audio ---")
            try:
                pl_module.eval()
                with torch.no_grad():
                    # Common synthesis methods
                    for method_name in ['infer', 'synthesize', 'generate', 'forward', 'inference']:
                        if hasattr(pl_module, method_name):
                            print(f"Model has method: {method_name}")
                pl_module.train()
            except Exception as e:
                print(f"Error: {e}")
            
            print("=== END DEBUG ===\n")

