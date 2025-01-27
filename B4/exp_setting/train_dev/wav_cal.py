import soundfile as sf
import os

def calculate_total_duration(scp_file):
    """
    SCPファイルに記載された音声ファイルの総時間を計算する関数
    
    Args:
        scp_file (str): SCPファイルのパス
    
    Returns:
        float: 総時間（秒）
    """
    total_duration = 0
    error_files = []
    
    try:
        with open(scp_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 空行やハイフンのみの行をスキップ
                if not line.strip() or line.strip().startswith('-'):
                    continue
                
                # ファイルパスを取得（2列目）
                try:
                    _, wav_path = line.strip().split(maxsplit=1)
                    
                    # ファイルが存在するか確認
                    if not os.path.exists(wav_path):
                        error_files.append(f"File not found: {wav_path}")
                        continue
                    
                    # 音声ファイルの情報を取得
                    with sf.SoundFile(wav_path) as wav_file:
                        duration = len(wav_file) / wav_file.samplerate
                        total_duration += duration
                        
                except Exception as e:
                    error_files.append(f"Error processing {wav_path}: {str(e)}")
                    
    except FileNotFoundError:
        print(f"Error: SCPファイル '{scp_file}' が見つかりません。")
        return None
    except Exception as e:
        print(f"Error: ファイル読み込み中にエラーが発生しました: {e}")
        return None
    
    return total_duration, error_files

def main():
    scp_file = 'wav.scp'
    
    # 総時間を計算
    result = calculate_total_duration(scp_file)
    if result is None:
        return
        
    total_duration, error_files = result
    
    # 結果をファイルに書き出す
    try:
        with open('all_cal.txt', 'w', encoding='utf-8') as f:
            f.write(f"総時間: {total_duration:.2f} 秒\n")
            f.write(f"総時間: {total_duration/60:.2f} 分\n\n")
            
            if error_files:
                f.write("エラーが発生したファイル:\n")
                for error in error_files:
                    f.write(f"{error}\n")
                    
        print("結果を all_cal.txt に書き出しました。")
    except Exception as e:
        print(f"Error: 結果の書き込み中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()