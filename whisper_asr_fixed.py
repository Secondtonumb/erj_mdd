import json
import torch
import os
import numpy as np
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import the correct whisper package
try:
    import whisper
    # Test if it's the correct whisper package
    if hasattr(whisper, 'load_model'):
        logger.info("Using OpenAI Whisper package")
    else:
        raise ImportError("Wrong whisper package")
except ImportError:
    logger.error("OpenAI Whisper not found. Please install it with:")
    logger.error("pip install openai-whisper")
    logger.error("Or: pip install git+https://github.com/openai/whisper.git")
    exit(1)

class WhisperASRProcessor:
    def __init__(self, model_name="medium", device=None):
        """
        Initialize Whisper ASR processor
        
        Args:
            model_name: Whisper model size ('tiny', 'base', 'small', 'medium', 'large')
            device: Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Load Whisper model
        logger.info(f"Loading Whisper model: {model_name}")
        try:
            self.model = whisper.load_model(model_name, device=self.device)
            logger.info("Whisper model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            logger.error("Please make sure you have the correct OpenAI Whisper package installed:")
            logger.error("pip install openai-whisper")
            raise
    
    def transcribe_audio(self, audio_path):
        """
        Transcribe a single audio file using Whisper
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Transcribed text
        """
        try:
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                return ""
            
            # Transcribe using Whisper
            result = self.model.transcribe(
                audio_path,
                language="en",  # Force English
                task="transcribe",
                fp16=False if self.device == 'cpu' else True,  # Use fp16 for GPU
                verbose=False
            )
            
            return result["text"].strip()
            
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
        
        logger.info(f"Processing {total_utterances} utterances with Whisper...")
        
        for i, (key, utterance_data) in enumerate(tqdm(data.items(), desc="Transcribing")):
            audio_path = utterance_data.get('wav', '')
            
            if not audio_path:
                logger.warning(f"No audio path found for utterance {key}")
                continue
            
            # Transcribe audio
            transcription = self.transcribe_audio(audio_path)
            transcription = transcription.upper()
            transcription = transcription.replace(".", "")
            transcription = transcription.replace(",", "")
            transcription = transcription.replace("!", "")
            transcription = transcription.replace("?", "")
            transcription = transcription.replace(":", "")
            transcription = transcription.replace(";", "")
            transcription = transcription.replace("(", "")
            transcription = transcription.replace(")", "")
            transcription = transcription.replace("\"", "")
            transcription = transcription.replace("\'", "")
            transcription = transcription.replace("\"", "")
            transcription = transcription.replace("\"", "")
            
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
        Calculate Word Error Rate (WER) for all transcriptions, and report
        substitution, deletion, and insertion statistics for each sentence and overall.

        Args:
            results: Dictionary containing transcription results

        Returns:
            dict: {
                'avg_wer': float,
                'total_S': int,
                'total_D': int,
                'total_I': int,
                'sentence_details': {
                    key: {
                        'wer': float,
                        'S': int,
                        'D': int,
                        'I': int,
                        'ref_len': int,
                        'reference': str,
                        'hypothesis': str
                    }, ...
                }
            }
        """
        def compute_wer_details(reference, hypothesis):
            """Compute WER and error counts (S, D, I) using dynamic programming."""
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            n = len(ref_words)
            m = len(hyp_words)

            # Distance and backtrace matrices
            d = np.zeros((n + 1, m + 1), dtype=int)
            op = np.zeros((n + 1, m + 1), dtype=int)  # 0: OK, 1: Sub, 2: Ins, 3: Del

            for i in range(n + 1):
                d[i, 0] = i
                if i > 0:
                    op[i, 0] = 3  # deletion
            for j in range(m + 1):
                d[0, j] = j
                if j > 0:
                    op[0, j] = 2  # insertion

            for i in range(1, n + 1):
                for j in range(1, m + 1):
                    if ref_words[i - 1] == hyp_words[j - 1]:
                        d[i, j] = d[i - 1, j - 1]
                        op[i, j] = 0  # correct
                    else:
                        sub = d[i - 1, j - 1] + 1
                        ins = d[i, j - 1] + 1
                        dele = d[i - 1, j] + 1
                        min_cost = min(sub, ins, dele)
                        d[i, j] = min_cost
                        if min_cost == sub:
                            op[i, j] = 1  # substitution
                        elif min_cost == ins:
                            op[i, j] = 2  # insertion
                        else:
                            op[i, j] = 3  # deletion

            # Backtrace to count S, D, I
            i, j = n, m
            S = D = I = 0
            while i > 0 or j > 0:
                if op[i, j] == 0:
                    i -= 1
                    j -= 1
                elif op[i, j] == 1:
                    S += 1
                    i -= 1
                    j -= 1
                elif op[i, j] == 2:
                    I += 1
                    j -= 1
                elif op[i, j] == 3:
                    D += 1
                    i -= 1

            wer = (S + D + I) / n if n > 0 else 1.0
            return wer, S, D, I, n

        total_wer = 0
        valid_transcriptions = 0
        total_S = 0
        total_D = 0
        total_I = 0
        sentence_details = {}

        for key, result in results.items():
            reference = result['reference_text']
            hypothesis = result['asr_transcription']

            if reference and hypothesis:
                wer, S, D, I, ref_len = compute_wer_details(reference, hypothesis)
                total_wer += wer
                total_S += S
                total_D += D
                total_I += I
                valid_transcriptions += 1
                sentence_details[key] = {
                    'wer': wer,
                    'S': S,
                    'D': D,
                    'I': I,
                    'ref_len': ref_len,
                    'reference': reference,
                    'hypothesis': hypothesis
                }
            else:
                sentence_details[key] = {
                    'wer': None,
                    'S': None,
                    'D': None,
                    'I': None,
                    'ref_len': None,
                    'reference': reference,
                    'hypothesis': hypothesis
                }

        avg_wer = total_wer / valid_transcriptions if valid_transcriptions > 0 else 0
        logger.info(f"Average WER: {avg_wer:.4f} ({valid_transcriptions} valid transcriptions)")
        logger.info(f"Total Substitutions: {total_S}, Deletions: {total_D}, Insertions: {total_I}")

        # Print per-sentence error details (optional: only first 10 for brevity)
        logger.info("Sample sentence-level WER and error details:")
        shown = 0
        for key, details in sentence_details.items():
            if details['wer'] is not None and shown < 10:
                logger.info(
                    f"Utterance: {os.path.basename(key)} | WER: {details['wer']:.3f} | "
                    f"S: {details['S']} D: {details['D']} I: {details['I']} | "
                    f"Ref: {details['reference']} | Hyp: {details['hypothesis']}"
                )
                shown += 1

        return {
            'avg_wer': avg_wer,
            'total_S': total_S,
            'total_D': total_D,
            'total_I': total_I,
            'sentence_details': sentence_details
        }
        total_wer = 0
        valid_transcriptions = 0
        
        for key, result in results.items():
            reference = result['reference_text']
            hypothesis = result['asr_transcription']
            import pdb; pdb.set_trace()
            
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
        logger.info(f"\n=== Sample Whisper Transcription Results (showing {num_samples} samples) ===")
        
        sample_count = 0
        for key, result in list(results.items())[:num_samples]:
            print(f"\nUtterance: {os.path.basename(key)}")
            print(f"Speaker: {result['speaker_id']}")
            print(f"Duration: {result['duration']:.2f}s")
            print(f"Reference: {result['reference_text']}")
            print(f"Whisper Output: {result['asr_transcription']}")
            print("-" * 80)
            sample_count += 1
            
            if sample_count >= num_samples:
                break

def main():
    # Configuration
    json_path = "/home/kevingenghaopeng/MDD/mpl-mdd/data/test_erj_spk_open_test_1.1.json"
    output_path = "/home/kevingenghaopeng/MDD/mpl-mdd/results/whisper_asr_results.json"
    
    # Initialize Whisper ASR processor
    asr_processor = WhisperASRProcessor(
        model_name="large-v3-turbo",  # Options: tiny, base, small, medium, large
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process all utterances
    results = asr_processor.process_utterances(json_path, output_path)
    
    # Calculate and display WER
    avg_wer = asr_processor.calculate_wer(results)
    
    # Print sample results
    asr_processor.print_sample_results(results, num_samples=5)
    
    logger.info("Whisper ASR processing completed!")

if __name__ == "__main__":
    main() 