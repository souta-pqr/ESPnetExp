import re
from collections import Counter

def extract_fillers(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    pattern = r'\(W (.+?)\)'
    matches = re.findall(pattern, text)
    
    # Count occurrences of each filler word
    filler_counts = Counter(matches)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for filler, count in filler_counts.items():
            f.write(f"{filler}\n")
        
        # Write total count of unique filler words
        f.write(f"\n総言い誤りーワード数: {len(filler_counts)}\n")
        
        # Write count of each filler word
        f.write("\n各言い誤りワードの出現回数:\n")
        for filler, count in filler_counts.items():
            f.write(f"{filler}: {count}回\n")

# 使用例
input_file = 'text'
output_file = 'output_speech_error.txt'
extract_fillers(input_file, output_file)