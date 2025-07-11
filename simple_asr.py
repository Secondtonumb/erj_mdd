import json
import torch
import torchaudio
import os
import numpy as np
from tqdm import tqdm
import logging
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleASRProcessor:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device=None):
        """
        Initialize simple ASR processor using HuggingFace transformers
        
        Args:
            model_name: Name of the wav2vec2 model to use
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load model and processor
        logger.info(f"Loading model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully!")
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return ""
            
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Prepare input
            inputs = self.processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = self.processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
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
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def calculate_wer(self, results):
        """
        Calculate Word Error Rate (WER) for all transcriptions
        
        Args:
            results: Dictionary containing transcription results
            
        Returns:
            float: Average WER
        """
        def compute_wer(reference, hypothesis):
            """Simple WER calculation"""
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            
            # Create distance matrix
            d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1))
            d[:, 0] = np.arange(len(ref_words) + 1)
            d[0, :] = np.arange(len(hyp_words) + 1)
            
            for i in range(1, len(ref_words) + 1):
                for j in range(1, len(hyp_words) + 1):
                    if ref_words[i-1] == hyp_words[j-1]:
                        d[i, j] = d[i-1, j-1]
                    else:
                        d[i, j] = min(d[i-1, j], d[i, j-1], d[i-1, j-1]) + 1
            
            return d[len(ref_words), len(hyp_words)] / len(ref_words) if len(ref_words) > 0 else 1.0
        
        total_wer = 0
        valid_transcriptions = 0
        
        for key, result in results.items():
            reference = result['reference_text']
            hypothesis = result['asr_transcription']
            
            if reference and hypothesis:
                # Calculate WER for this utterance
                wer = compute_wer(reference, hypothesis)
                total_wer += wer
                valid_transcriptions += 1
        
        avg_wer = total_wer / valid_transcriptions if valid_transcriptions > 0 else 0
        logger.info(f"Average WER: {avg_wer:.4f} ({valid_transcriptions} valid transcriptions)")
        
        return avg_wer
    
    def print_sample_results(self, results, num_samples=5):
        """
        Print sample transcription results
        
        Args:
            results: Dictionary containing transcription results
            num_samples: Number of samples to print
        """
        logger.info(f"\n=== Sample Transcription Results (showing {num_samples} samples) ===")
        
        sample_count = 0
        for key, result in list(results.items())[:num_samples]:
            print(f"\nUtterance: {os.path.basename(key)}")
            print(f"Speaker: {result['speaker_id']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Reference: {result['reference_text']}")
            print(f"ASR Output: {result['asr_transcription']}")
            print("-" * 80)
            sample_count += 1
            
            if sample_count >= num_samples:
                break

def main():
    # Configuration
    json_path = "/home/kevingenghaopeng/MDD/mpl-mdd/data/test_erj_spk_open_test_1.1.json"
    output_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/simple_asr_results.json"
    
    # Initialize ASR processor
    asr_processor = SimpleASRProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process all utterances
    results = asr_processor.process_utterances(json_path, output_path)
    
    # Calculate and display WER
    avg_wer = asr_processor.calculate_wer(results)
    
    # Print sample results
    asr_processor.print_sample_results(results, num_samples=5)
    
    logger.info("ASR processing completed!")

if __name__ == "__main__":
    main() 