import sys
from pathlib import Path
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
import numpy as np
import wave
import yaml
import time
from datetime import datetime
import pyaudio
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class WordTimestamp:
    """単語とそのタイムスタンプを保持するクラス"""
    word: str
    audio_time: float  # 音声内での時刻
    process_time: float  # システム処理時刻
    latency: float  # 遅延時間（process_time - audio_time）

@dataclass
class SPLMeasurement:
    """SPLの測定結果を保持するクラス"""
    audio_duration: float  # 音声の実際の長さ（秒）
    final_latency: float  # 最後の単語の認識遅延（ミリ秒）
    all_words: List[WordTimestamp]  # すべての単語の認識タイミング
    final_text: str  # 最終的な認識テキスト

class StreamingSpeechRecognizer:
    def __init__(self, config_path: str, model_path: str):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        self.speech2text = Speech2TextStreaming(
            asr_train_config=config_path,
            asr_model_file=model_path,
            token_type=None,
            bpemodel=None,
            maxlenratio=0.0,
            minlenratio=0.0,
            beam_size=20,
            ctc_weight=0.5,
            lm_weight=0.0,
            penalty=0.0,
            nbest=1,
            device="cpu",
            disable_repetition_detection=True,
            decoder_text_length_limit=0,
            encoded_feat_length_limit=0
        )
        
        self.chunk_size = 640  # 16kHz * 0.04s = 640 samples

    def _get_new_words(self, prev_text: str, curr_text: str) -> List[str]:
        """前回の認識テキストと現在の認識テキストを比較して、新しい単語を抽出"""
        if not prev_text:
            return curr_text.split()
        
        prev_words = prev_text.split()
        curr_words = curr_text.split()
        
        if len(curr_words) <= len(prev_words):
            return []
            
        return curr_words[len(prev_words):]

    def measure_spl(self, wav_path: str, verbose: bool = True) -> SPLMeasurement:
        """改良版SPL測定関数"""
        # 音声ファイルの読み込みと音声長の計算
        with wave.open(wav_path, 'rb') as wav_file:
            rate = wav_file.getframerate()
            nframes = wav_file.getnframes()
            audio_duration = nframes / rate
            audio_data = wav_file.readframes(nframes)
            
        speech = np.frombuffer(audio_data, dtype='int16').astype(np.float16) / 32767.0
        
        if verbose:
            print(f"\nProcessing audio file: {wav_path}")
            print(f"Audio duration: {audio_duration:.2f}s")
            print("\nRecognition Progress:")
            print("Audio Time | Process Time | Latency | Word")
            print("-" * 70)
        
        start_time = time.time()
        previous_text = ""
        word_timestamps: List[WordTimestamp] = []
        
        for i in range(0, len(speech), self.chunk_size):
            chunk = speech[i:i + self.chunk_size]
            audio_time = i / rate
            
            is_final = (i + self.chunk_size >= len(speech))
            results = self.speech2text(speech=chunk, is_final=is_final)
            
            if results and len(results) > 0:
                current_text = results[0][0]
                
                if current_text and current_text != previous_text:
                    process_time = time.time() - start_time
                    new_words = self._get_new_words(previous_text, current_text)
                    
                    for word in new_words:
                        latency = process_time - audio_time
                        word_timestamp = WordTimestamp(
                            word=word,
                            audio_time=audio_time,
                            process_time=process_time,
                            latency=latency
                        )
                        word_timestamps.append(word_timestamp)
                        
                        if verbose:
                            print(f"{audio_time:>10.2f}s | {process_time:>12.2f}s | {latency*1000:>7.1f}ms | {word}")
                    
                    previous_text = current_text
        
        # 最後の単語の遅延をSPLとする
        final_latency = word_timestamps[-1].latency * 1000 if word_timestamps else 0
        
        measurement = SPLMeasurement(
            audio_duration=audio_duration,
            final_latency=final_latency,
            all_words=word_timestamps,
            final_text=previous_text
        )
        
        if verbose:
            print("\nSPL Measurement Results:")
            print(f"Audio Duration: {measurement.audio_duration:.2f}s")
            print(f"Final Word Latency (SPL): {measurement.final_latency:.2f}ms")
            print(f"Total Words Processed: {len(measurement.all_words)}")
            print("\nLatency Statistics:")
            latencies = [wt.latency * 1000 for wt in measurement.all_words]
            print(f"Average Latency: {sum(latencies)/len(latencies):.2f}ms")
            print(f"Min Latency: {min(latencies):.2f}ms")
            print(f"Max Latency: {max(latencies):.2f}ms")
            
        return measurement

def main():
    """メイン関数"""
    data_dir = Path("data")
    config_path = str(data_dir / "config.yaml")
    model_path = str(data_dir / "valid.acc.best.pth")
    
    recognizer = StreamingSpeechRecognizer(config_path, model_path)
    
    wav_path = "sample.wav"
    print(f"\nMeasuring SPL for file: {wav_path}")
    spl_result = recognizer.measure_spl(wav_path)

if __name__ == "__main__":
    main()