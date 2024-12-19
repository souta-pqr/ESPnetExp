import re

def merge_f_markers(text):
    """
    Merges consecutive (F) text1 (/F) (F) text2 (/F) patterns into a single (F) text1 text2 (/F) pattern.
    
    Args:
        text (str): Input text containing (F) text (/F) patterns
        
    Returns:
        str: Text with merged consecutive F-tagged patterns
    """
    # Split text into lines
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        # Find all (F) text (/F) patterns
        while True:
            # Pattern to match consecutive (F) text (/F) (F) text (/F)
            pattern = r'\(I\)\s*([^(]*?)\s*\(/I\)\s*\(I\)\s*([^(]*?)\s*\(/I\)'
            match = re.search(pattern, line)
            
            if not match:
                break
                
            # Merge the two matches into one
            # Remove trailing and leading spaces from captured groups
            text1 = match.group(1).strip()
            text2 = match.group(2).strip()
            merged = f'(I) {text1} {text2} (/I)'
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