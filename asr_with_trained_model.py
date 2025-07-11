import json
import torch
import torchaudio
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import os
import numpy as np
from tqdm import tqdm
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainedASRProcessor:
    def __init__(self, hparams_path, checkpoint_path=None, device=None):
        """
        Initialize ASR processor with trained model
        
        Args:
            hparams_path: Path to YAML hyperparameters file
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load hyperparameters
        with open(hparams_path, 'r') as f:
            self.hparams = yaml.safe_load(f)
        
        # Initialize model from hyperparameters
        self.model = self._load_model_from_hparams()
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
        
        self.model.to(self.device)
        self.model.eval()
    
    def _load_model_from_hparams(self):
        """
        Load model architecture from hyperparameters
        """
        # This is a simplified version - you might need to adapt based on your specific model structure
        from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
        from speechbrain.lobes.models.VanillaNN import VanillaNN
        from speechbrain.nnet.linear import Linear
        
        # Load wav2vec2
        wav2vec2 = HuggingFaceWav2Vec2(
            source=self.hparams.get('wav2vec2_hub', 'facebook/wav2vec2-base'),
            output_norm=True,
            freeze=self.hparams.get('freeze_wav2vec', False),
            freeze_feature_extractor=self.hparams.get('freeze_wav2vec_feature_extractor', True),
            save_path=self.hparams.get('save_folder', './save') + '/wav2vec2_checkpoint'
        )
        
        # Load encoder
        enc = VanillaNN(
            input_shape=[None, None, 768],
            activation=torch.nn.LeakyReLU(),
            dnn_blocks=self.hparams.get('dnn_layers', 2),
            dnn_neurons=self.hparams.get('dnn_neurons', 384)
        )
        
        # Load CTC linear layer
        ctc_lin = Linear(
            input_size=self.hparams.get('dnn_neurons', 384),
            n_neurons=self.hparams.get('output_neurons', 42)
        )
        
        # Create model
        model = torch.nn.ModuleList([enc, ctc_lin])
        
        return model
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file using the trained model
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return ""
            
            # Load and preprocess audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Resample if necessary
            if sample_rate != self.hparams.get('sample_rate', 16000):
                resampler = torchaudio.transforms.Resample(sample_rate, self.hparams.get('sample_rate', 16000))
                waveform = resampler(waveform)
            
            # Move to device
            waveform = waveform.to(self.device)
            
            # Forward pass through model
            with torch.no_grad():
                # This is a simplified forward pass - you'll need to adapt based on your model structure
                features = self.model[0](waveform)  # Encoder
                logits = self.model[1](features)    # CTC linear
                
                # Apply log softmax
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # Decode (simplified - you might want to use a proper CTC decoder)
                # For now, just return a placeholder
                transcription = "placeholder_transcription"
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {str(e)}")
            return ""
    
    def process_utterances(self, json_path, output_path=None):
        """
        Process all utterances in the JSON file
        
        Args:
            json_path: Path to JSON file with utterance data
            output_path: Path to save results (optional)
        """
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        total_utterances = len(data)
        
        logger.info(f"Processing {total_utterances} utterances...")
        
        for i, (key, utterance_data) in enumerate(tqdm(data.items(), desc="Transcribing")):
            audio_path = utterance_data.get('wav', '')
            
            if not audio_path:
                logger.warning(f"No audio path found for utterance {key}")
                continue
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            
            # Store results
            results[key] = {
                'original': utterance_data,
                'asr_transcription': transcription,
                'reference_text': utterance_data.get('wrd', '').strip(),
                'speaker_id': utterance_data.get('spk_id', ''),
                'duration': utterance_data.get('duration', 0)
            }
            
            # Log progress every 10 utterances
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total_utterances} utterances")
        
        # Save results
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results

def main():
    # Configuration
    json_path = "/home/kevingenghaopeng/MDD/mpl-mdd/data/test_erj_spk_open_test_1.1.json"
    hparams_path = "/home/kevingenghaopeng/MDD/mpl-mdd/hparams/evaluate_erj_mpl.yaml"
    output_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/trained_asr_results.json"
    
    # Optional: Path to your trained model checkpoint
    # checkpoint_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/wav2vec2-base_ctc_erj_new/save/model.ckpt"
    
    # Initialize ASR processor with trained model
    asr_processor = TrainedASRProcessor(
        hparams_path=hparams_path,
        # checkpoint_path=checkpoint_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process all utterances
    results = asr_processor.process_utterances(json_path, output_path)
    
    logger.info("ASR processing completed!")

if __name__ == "__main__":
    main() 