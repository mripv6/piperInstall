import torch
import os
from pathlib import Path
from lightning.pytorch.callbacks import Callback


class AudioLoggerCallback(Callback):
    """
    Callback to log intermediate audio samples to TensorBoard during Piper TTS training.
    
    Args:
        log_every_n_steps: How often to log audio (in training steps)
        sample_rate: Audio sample rate (Piper typically uses 22050 Hz)
        max_audio_length: Maximum audio length to log in seconds
        validation_text: Optional fixed text to synthesize for consistent comparison
    """
    
    def __init__(
        self, 
        log_every_n_steps: int = 1000,
        sample_rate: int = 22050,
        max_audio_length: float = 10.0,
        validation_text: str = None,
        save_to_disk: bool = True,
        output_dir: str = "audio_samples"
    ):
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_length * sample_rate)
        self.validation_text = validation_text
        self.save_to_disk = save_to_disk
        self.output_dir = output_dir
        
        # Create output directory if saving to disk
        if self.save_to_disk:
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def on_train_batch_end(
        self, 
        trainer, 
        pl_module, 
        outputs, 
        batch, 
        batch_idx
    ):
        """Log audio during training at specified intervals"""
        # Only log at specified intervals
        if trainer.global_step % self.log_every_n_steps != 0:
            return
        
        # Skip if this is step 0
        if trainer.global_step == 0:
            return
        
        self._log_audio_sample(trainer, pl_module, batch, outputs)
    
    def _log_audio_sample(self, trainer, pl_module, batch, outputs):
        """Generate and log audio to TensorBoard"""
        try:
            # Put model in eval mode
            was_training = pl_module.training
            pl_module.eval()
            
            with torch.no_grad():
                # Extract audio from outputs or batch
                # The exact method depends on Piper's model structure
                audio_output = self._extract_audio(outputs, batch, pl_module)
                
                if audio_output is None:
                    print(f"[AudioLogger] Could not extract audio at step {trainer.global_step}")
                    return
                
                # Ensure audio is on CPU
                audio_output = audio_output.cpu()
                
                # Ensure correct shape: (channels, samples) or (samples,)
                if audio_output.dim() == 1:
                    audio_output = audio_output.unsqueeze(0)  # Add channel dimension
                elif audio_output.dim() > 2:
                    # If batch dimension exists, take first sample
                    audio_output = audio_output[0]
                
                # Truncate if too long
                if audio_output.shape[-1] > self.max_samples:
                    audio_output = audio_output[..., :self.max_samples]
                
                # Normalize to [-1, 1] range
                max_val = audio_output.abs().max()
                if max_val > 0:
                    audio_output = audio_output / max_val
                
                # Log to TensorBoard
                if trainer.logger:
                    try:
                        trainer.logger.experiment.add_audio(
                            'training/generated_audio',
                            audio_output,
                            trainer.global_step,
                            sample_rate=self.sample_rate
                        )
                        print(f"[AudioLogger] Logged audio to TensorBoard at step {trainer.global_step}")
                    except Exception as e:
                        print(f"[AudioLogger] Failed to log to TensorBoard: {e}")
                
                # Save to disk as WAV file
                if self.save_to_disk:
                    try:
                        import torchaudio
                        import warnings
                        output_path = Path(self.output_dir) / f"step_{trainer.global_step:08d}.wav"
                        
                        # Suppress torchaudio deprecation warnings
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning, module='torchaudio')
                            torchaudio.save(
                                str(output_path),
                                audio_output.cpu(),
                                self.sample_rate
                            )
                        print(f"[AudioLogger] Saved audio to {output_path}")
                    except ImportError:
                        print(f"[AudioLogger] torchaudio not installed, skipping disk save. Install with: pip install torchaudio")
                    except Exception as e:
                        print(f"[AudioLogger] Failed to save audio to disk: {e}")
            
            # Restore training mode
            if was_training:
                pl_module.train()
                
        except Exception as e:
            print(f"[AudioLogger] Audio logging failed at step {trainer.global_step}: {e}")
            import traceback
            traceback.print_exc()
    
    def _extract_audio(self, outputs, batch, pl_module):
        """
        Extract audio from model outputs or batch.
        
        For Piper VITS training, the batch object has:
        - batch.audios: ground truth audio waveforms
        - batch.audio_lengths: actual lengths of each audio
        - batch.spectrograms: mel spectrograms
        """
        
        # Try to extract from outputs first (generated audio)
        if outputs is not None and isinstance(outputs, dict):
            for key in ['audio', 'wav', 'waveform', 'audio_output', 'y_hat']:
                if key in outputs and outputs[key] is not None:
                    audio = outputs[key]
                    if isinstance(audio, torch.Tensor):
                        # Take first sample if it's a batch
                        if audio.dim() > 1 and audio.shape[0] > 1:
                            return audio[0]
                        return audio
        
        # Fall back to ground truth audio from batch (Piper VITS format)
        if batch is not None:
            # Check if it's a Piper Batch object with 'audios' attribute
            if hasattr(batch, 'audios'):
                audio = batch.audios
                audio_length = batch.audio_lengths[0] if hasattr(batch, 'audio_lengths') else None
                
                # Take first sample from batch
                if audio.dim() > 1:
                    audio = audio[0]
                
                # Trim to actual length if available
                if audio_length is not None and audio_length > 0:
                    audio = audio[:audio_length]
                
                return audio
            
            # Fallback: try dict-like batch
            elif isinstance(batch, dict):
                for key in ['audio', 'audios', 'wav', 'waveform', 'y']:
                    if key in batch:
                        audio = batch[key]
                        if isinstance(audio, torch.Tensor):
                            if audio.dim() > 1 and audio.shape[0] > 1:
                                return audio[0]
                            return audio
            
            # Fallback: try tuple/list batch
            elif isinstance(batch, (tuple, list)) and len(batch) >= 2:
                audio = batch[-1]
                if isinstance(audio, torch.Tensor):
                    if audio.dim() > 1 and audio.shape[0] > 1:
                        return audio[0]
                    return audio
        
        return None


class MultiTextAudioLoggerCallback(AudioLoggerCallback):
    """
    Extended version that logs multiple text samples for better comparison.
    """
    
    def __init__(
        self,
        validation_texts: list = None,
        log_every_n_steps: int = 1000,
        sample_rate: int = 22050,
        max_audio_length: float = 10.0,
        save_to_disk: bool = True,
        output_dir: str = "audio_samples"
    ):
        super().__init__(
            log_every_n_steps=log_every_n_steps,
            sample_rate=sample_rate,
            max_audio_length=max_audio_length,
            save_to_disk=save_to_disk,
            output_dir=output_dir
        )
        
        # Default validation texts for TTS
        self.validation_texts = validation_texts or [
            "The quick brown fox jumps over the lazy dog.",
            "Hello, this is a test of the text to speech system.",
            "How are you doing today?"
        ]
    
    def _log_audio_sample(self, trainer, pl_module, batch, outputs):
        """Log multiple audio samples for different texts"""
        try:
            was_training = pl_module.training
            pl_module.eval()
            
            with torch.no_grad():
                # First, log the audio from the current batch
                super()._log_audio_sample(trainer, pl_module, batch, outputs)
                
                # Then, try to synthesize custom validation texts
                # This requires knowing Piper's synthesis API
                # Uncomment and adapt based on Piper's model structure:
                
                for idx, text in enumerate(self.validation_texts):
                    try:
                        audio = pl_module.synthesize(text)  # Adapt method name
                        audio = audio.cpu()
                        
                        if audio.dim() == 1:
                            audio = audio.unsqueeze(0)
                        
                        if audio.shape[-1] > self.max_samples:
                            audio = audio[..., :self.max_samples]
                        
                        max_val = audio.abs().max()
                        if max_val > 0:
                            audio = audio / max_val
                        
                        if trainer.logger:
                            trainer.logger.experiment.add_audio(
                                f'validation/sample_{idx}',
                                audio,
                                trainer.global_step,
                                sample_rate=self.sample_rate
                            )
                    except Exception as e:
                        print(f"[AudioLogger] Failed to synthesize text {idx}: {e}")
            
            if was_training:
                pl_module.train()
                
        except Exception as e:
            print(f"[AudioLogger] Multi-text audio logging failed: {e}")

