import json
import torch
import os
import numpy as np
from tqdm import tqdm
import logging
import whisper
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRComparison:
    def __init__(self, device=None):
        """
        Initialize ASR comparison with both Whisper and wav2vec2 models
        
        Args:
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model
        logger.info("Loading Whisper medium model...")
        self.whisper_model = whisper.load_model("medium", device=self.device)
        
        # Load wav2vec2 model
        logger.info("Loading wav2vec2 model...")
        self.wav2vec2_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.wav2vec2_model.to(self.device)
        self.wav2vec2_model.eval()
        
        logger.info("Both models loaded successfully!")
    
    def transcribe_with_whisper(self, audio_path):
        """Transcribe using Whisper"""
        try:
            if not os.path.exists(audio_path):
                return ""
            
            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                task="transcribe",
                fp16=False if self.device == 'cpu' else True,
                verbose=False
            )
            
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Whisper error for {audio_path}: {str(e)}")
            return ""
    
    def transcribe_with_wav2vec2(self, audio_path):
        """Transcribe using wav2vec2"""
        try:
            if not os.path.exists(audio_path):
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
            inputs = self.wav2vec2_processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = self.wav2vec2_model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = self.wav2vec2_processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"wav2vec2 error for {audio_path}: {str(e)}")
            return ""
    
    def compute_wer(self, reference, hypothesis):
        """Compute Word Error Rate"""
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
    
    def process_utterances(self, json_path, output_path=None):
        """
        Process all utterances with both models
        
        Args:
            json_path: Path to JSON file with utterance data
            output_path: Path to save results (optional)
        """
        # Load JSON data
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {}
        total_utterances = len(data)
        
        logger.info(f"Processing {total_utterances} utterances with both models...")
        
        whisper_wer_total = 0
        wav2vec2_wer_total = 0
        valid_transcriptions = 0
        
        for i, (key, utterance_data) in enumerate(tqdm(data.items(), desc="Transcribing")):
            audio_path = utterance_data.get('wav', '')
            
            if not audio_path:
                logger.warning(f"No audio path found for utterance {key}")
                continue
            
            # Transcribe with both models
            whisper_transcription = self.transcribe_with_whisper(audio_path)
            wav2vec2_transcription = self.transcribe_with_wav2vec2(audio_path)
            
            reference_text = utterance_data.get('wrd', '').strip()
            
            # Calculate WER for both models
            whisper_wer = 0
            wav2vec2_wer = 0
            
            if reference_text and whisper_transcription:
                whisper_wer = self.compute_wer(reference_text, whisper_transcription)
                whisper_wer_total += whisper_wer
            
            if reference_text and wav2vec2_transcription:
                wav2vec2_wer = self.compute_wer(reference_text, wav2vec2_transcription)
                wav2vec2_wer_total += wav2vec2_wer
            
            if reference_text and (whisper_transcription or wav2vec2_transcription):
                valid_transcriptions += 1
            
            # Store results
            results[key] = {
                'original': utterance_data,
                'whisper_transcription': whisper_transcription,
                'wav2vec2_transcription': wav2vec2_transcription,
                'reference_text': reference_text,
                'speaker_id': utterance_data.get('spk_id', ''),
                'duration': utterance_data.get('duration', 0),
                'whisper_wer': whisper_wer,
                'wav2vec2_wer': wav2vec2_wer
            }
            
            # Log progress every 10 utterances
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{total_utterances} utterances")
        
        # Calculate average WER
        avg_whisper_wer = whisper_wer_total / valid_transcriptions if valid_transcriptions > 0 else 0
        avg_wav2vec2_wer = wav2vec2_wer_total / valid_transcriptions if valid_transcriptions > 0 else 0
        
        logger.info(f"\n=== ASR Model Comparison Results ===")
        logger.info(f"Whisper Medium Average WER: {avg_whisper_wer:.4f}")
        logger.info(f"wav2vec2-base Average WER: {avg_wav2vec2_wer:.4f}")
        logger.info(f"Valid transcriptions: {valid_transcriptions}")
        
        if avg_whisper_wer < avg_wav2vec2_wer:
            logger.info("Whisper performed better!")
        elif avg_wav2vec2_wer < avg_whisper_wer:
            logger.info("wav2vec2 performed better!")
        else:
            logger.info("Both models performed similarly!")
        
        # Save results
        if output_path:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Add summary to results
            results['_summary'] = {
                'avg_whisper_wer': avg_whisper_wer,
                'avg_wav2vec2_wer': avg_wav2vec2_wer,
                'valid_transcriptions': valid_transcriptions,
                'total_utterances': total_utterances
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {output_path}")
        
        return results
    
    def print_sample_results(self, results, num_samples=5):
        """
        Print sample transcription results from both models
        
        Args:
            results: Dictionary containing transcription results
            num_samples: Number of samples to print
        """
        logger.info(f"\n=== Sample Comparison Results (showing {num_samples} samples) ===")
        
        sample_count = 0
        for key, result in list(results.items())[:num_samples]:
            if key == '_summary':
                continue
                
            print(f"\nUtterance: {os.path.basename(key)}")
            print(f"Speaker: {result['speaker_id']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Reference: {result['reference_text']}")
            print(f"Whisper: {result['whisper_transcription']} (WER: {result['whisper_wer']:.3f})")
            print(f"wav2vec2: {result['wav2vec2_transcription']} (WER: {result['wav2vec2_wer']:.3f})")
            print("-" * 80)
            sample_count += 1
            
            if sample_count >= num_samples:
                break

def main():
    # Configuration
    json_path = "/home/kevingenghaopeng/MDD/mpl-mdd/data/test_erj_spk_open_test_1.1.json"
    output_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/asr_comparison_results.json"
    
    # Initialize ASR comparison
    asr_comparison = ASRComparison(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process all utterances with both models
    results = asr_comparison.process_utterances(json_path, output_path)
    
    # Print sample results
    asr_comparison.print_sample_results(results, num_samples=5)
    
    logger.info("ASR model comparison completed!")

if __name__ == "__main__":
    main() 