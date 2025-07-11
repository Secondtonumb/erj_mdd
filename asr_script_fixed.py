import json
import torch
import torchaudio
import speechbrain
from speechbrain.pretrained import EncoderDecoderASR
import os
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ASRProcessor:
    def __init__(self, model_path=None, device=None):
        """
        Initialize ASR processor
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize ASR model
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading custom model from: {model_path}")
            self.asr_model = EncoderDecoderASR.from_hparams(
                source=model_path,
                savedir=os.path.join(model_path, "pretrained_model"),
                run_opts={"device": self.device}
            )
        else:
            logger.info("Using pretrained wav2vec2 ASR model")
            # Use a different pretrained model that's more compatible
            try:
                self.asr_model = EncoderDecoderASR.from_hparams(
                    source="speechbrain/asr-crdnn-rnnlm-librispeech",
                    savedir="pretrained_models/asr-crdnn-rnnlm-librispeech",
                    run_opts={"device": self.device}
                )
            except Exception as e:
                logger.warning(f"Could not load SpeechBrain model: {e}")
                logger.info("Falling back to simple wav2vec2 approach")
                self.asr_model = None
    
    def transcribe_audio_simple(self, audio_path):
        """
        Simple transcription using wav2vec2 directly
        
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
            
            # Resample to 16kHz if needed
            if sample_rate != 16000:
                resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
            
            # Use wav2vec2 directly
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            
            # Load model and processor
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            
            model.to(self.device)
            model.eval()
            
            # Prepare input
            inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get logits
            with torch.no_grad():
                logits = model(**inputs).logits
            
            # Get predicted ids
            predicted_ids = torch.argmax(logits, dim=-1)
            
            # Decode
            transcription = processor.batch_decode(predicted_ids)[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {str(e)}")
            return ""
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        if self.asr_model is not None:
            try:
                if not os.path.exists(audio_path):
                    logger.warning(f"Audio file not found: {audio_path}")
                    return ""
                
                # Transcribe using the ASR model
                transcription = self.asr_model.transcribe_file(audio_path)
                return transcription.strip()
                
            except Exception as e:
                logger.error(f"Error with SpeechBrain model: {e}")
                logger.info("Falling back to simple transcription")
                return self.transcribe_audio_simple(audio_path)
        else:
            return self.transcribe_audio_simple(audio_path)
    
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
    
    def calculate_wer(self, results):
        """
        Calculate Word Error Rate (WER) for all transcriptions
        
        Args:
            results: Dictionary containing transcription results
            
        Returns:
            float: Average WER
        """
        from speechbrain.utils.metric_stats import ErrorRateStats
        
        wer_stats = ErrorRateStats()
        total_wer = 0
        valid_transcriptions = 0
        
        for key, result in results.items():
            reference = result['reference_text']
            hypothesis = result['asr_transcription']
            
            if reference and hypothesis:
                # Calculate WER for this utterance
                wer = wer_stats.compute_wer(reference, hypothesis)
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
    output_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/asr_results.json"
    
    # Optional: Path to your trained model (if you have one)
    # model_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/wav2vec2-base_ctc_erj_new/save"
    
    # Initialize ASR processor
    asr_processor = ASRProcessor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Process all utterances
    results = asr_processor.process_utterances(json_path, output_path)
    
    # Calculate and display WER
    avg_wer = asr_processor.calculate_wer(results)
    
    # Print sample results
    asr_processor.print_sample_results(results, num_samples=5)
    
    logger.info("ASR processing completed!")

if __name__ == "__main__":
    main() 