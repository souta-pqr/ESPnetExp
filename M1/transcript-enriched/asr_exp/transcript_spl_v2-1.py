import sys
from pathlib import Path
import espnet
from espnet2.bin.asr_inference_streaming import Speech2TextStreaming
from espnet_model_zoo.downloader import ModelDownloader
import numpy as np
import wave
import yaml
import time
from datetime import datetime
import pyaudio
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass

@dataclass
class SPLMeasurement:
    """SPLの測定結果を保持するクラス"""
    audio_duration: float  # 音声の実際の長さ（秒）
    process_duration: float  # 処理にかかった総時間（秒）
    spl: float  # Speech Perceived Latency（ミリ秒）
    last_text: str  # 最終的な認識テキスト

class StreamingSpeechRecognizer:
    """ストリーミング音声認識器のクラス"""
    def __init__(self, config_path: str, model_path: str):
        # ESPnetの設定
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Speech2Textの初期化
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
        
        self.chunk_size = 640  # 16kHz * 0.04s = 640 samples per chunk

    def measure_spl(self, wav_path: str, verbose: bool = True) -> SPLMeasurement:
        """SPLを測定する関数"""
        # 処理開始時刻を記録
        start_process_time = time.time()
        
        # 音声データの読み込みと音声長の計算
        with wave.open(wav_path, 'rb') as wav_file:
            rate = wav_file.getframerate()
            nframes = wav_file.getnframes()
            audio_duration = nframes / rate  # 音声の実際の長さ（秒）
            audio_data = wav_file.readframes(nframes)
            
        # 音声データの変換
        speech = np.frombuffer(audio_data, dtype='int16').astype(np.float16) / 32767.0
        
        if verbose:
            print(f"\nProcessing audio file: {wav_path}")
            print(f"Audio duration: {audio_duration:.2f}s")
            print("\nRecognition Progress:")
            print("Time | Current Text")
            print("-" * 50)
        
        current_text = ""
        
        # チャンクごとに処理
        for i in range(0, len(speech), self.chunk_size):
            chunk = speech[i:i + self.chunk_size]
            current_time = i / rate
            
            # 最後のチャンクかどうか
            is_final = (i + self.chunk_size >= len(speech))
            
            # 音声認識
            results = self.speech2text(
                speech=chunk,
                is_final=is_final
            )
            
            if results and len(results) > 0:
                text = results[0][0]  # 最良の認識結果を取得
                
                if text and text != current_text:
                    current_text = text
                    if verbose:
                        print(f"{current_time:.2f}s | {text}")
        
        # 処理終了時刻を記録
        end_process_time = time.time()
        process_duration = end_process_time - start_process_time
        
        # SPLを計算（実際の処理時間 - 音声の長さ）
        spl = (process_duration - audio_duration) * 1000  # ミリ秒に変換
        
        measurement = SPLMeasurement(
            audio_duration=audio_duration,
            process_duration=process_duration,
            spl=spl,
            last_text=current_text
        )
        
        if verbose:
            print("\nSPL Measurement Results:")
            print(f"Audio Duration: {measurement.audio_duration:.2f}s")
            print(f"Process Duration: {measurement.process_duration:.2f}s")
            print(f"SPL: {measurement.spl:.2f}ms")
            print(f"Final Recognition: {measurement.last_text}")
            
        return measurement

    def measure_realtime_spl(self, duration: int = 5, verbose: bool = True) -> SPLMeasurement:
        """リアルタイムでSPLを測定する関数"""
        CHUNK = 2048
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        
        p = pyaudio.PyAudio()
        stream = p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        start_process_time = time.time()
        current_text = ""
        audio_duration = duration  # 指定された録音時間
        
        if verbose:
            print("\nRealtime SPL Measurement:")
            print("Time | Current Text")
            print("-" * 50)
        
        try:
            for i in range(0, int(RATE/CHUNK * duration)):
                current_time = i * CHUNK / RATE
                
                # 音声データの読み取り
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype='int16')
                audio_data = audio_data.astype(np.float16) / 32767.0
                
                # 最後のチャンクかどうか
                is_final = (i == int(RATE/CHUNK * duration) - 1)
                
                # 音声認識
                results = self.speech2text(
                    speech=audio_data,
                    is_final=is_final
                )
                
                if results and len(results) > 0:
                    text = results[0][0]
                    
                    if text and text != current_text:
                        current_text = text
                        if verbose:
                            print(f"{current_time:.2f}s | {text}")
                            
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
        
        end_process_time = time.time()
        process_duration = end_process_time - start_process_time
        spl = (process_duration - audio_duration) * 1000
        
        measurement = SPLMeasurement(
            audio_duration=audio_duration,
            process_duration=process_duration,
            spl=spl,
            last_text=current_text
        )
        
        if verbose:
            print("\nRealtime SPL Measurement Results:")
            print(f"Audio Duration: {measurement.audio_duration:.2f}s")
            print(f"Process Duration: {measurement.process_duration:.2f}s")
            print(f"SPL: {measurement.spl:.2f}ms")
            print(f"Final Recognition: {measurement.last_text}")
            
        return measurement

def main():
    """メイン関数"""
    # パスの設定
    data_dir = Path("data")
    config_path = str(data_dir / "config.yaml")
    model_path = str(data_dir / "valid.acc.best.pth")
    
    # 認識器の初期化
    recognizer = StreamingSpeechRecognizer(config_path, model_path)
    
    # ファイルからのSPL測定
    wav_path = "sample.wav"
    print(f"\nMeasuring SPL for file: {wav_path}")
    file_spl = recognizer.measure_spl(wav_path)
    
    # # リアルタイムSPL測定を行う場合はコメントを解除(そのままだとエラーになる．コード変更する必要あり)
    # print("\nMeasuring realtime SPL (5 seconds)...")
    # realtime_spl = recognizer.measure_realtime_spl(duration=5)

if __name__ == "__main__":
    main()