import re

def merge_f_markers(text):
    """
    Merges consecutive (F text) patterns into a single (F text1 text2) pattern.
    
    Args:
        text (str): Input text containing (F text) patterns
        
    Returns:
        str: Text with merged consecutive (F text) patterns
    """
    # Split text into lines
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Find all (F ...) patterns
        while True:
            # Pattern to match consecutive (F ...) (F ...)
            pattern = r'\(D ([^)]+)\)\s*\(D ([^)]+)\)'
            match = re.search(pattern, line)
            
            if not match:
                break
                
            # Merge the two matches into one
            merged = f'(D {match.group(1)} {match.group(2)})'
            line = line[:match.start()] + merged + line[match.end():]
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)

# Read input file
try:
    with open('text', 'r', encoding='utf-8') as file:
        input_text = file.read()

    # Process the text
    processed_text = merge_f_markers(input_text)

    # Write output file
    with open('changed_text', 'w', encoding='utf-8') as file:
        file.write(processed_text)
        
    print("処理が完了しました。結果はchanged_textファイルに保存されています。")
    
except FileNotFoundError:
    print("入力ファイル 'text' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {str(e)}")