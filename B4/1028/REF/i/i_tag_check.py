def count_i_tags(line):
    """
    Count the number of (I) and (/I) tags in a line
    
    Args:
        line (str): Input line of text
        
    Returns:
        tuple: Count of opening and closing I tags
    """
    opening_count = line.count('(I)')
    closing_count = line.count('(/I)')
    return opening_count, closing_count

try:
    # Read input file and process lines
    multiple_tag_lines = []
    
    with open('text', 'r', encoding='utf-8') as file:
        for line in file:
            opening_count, closing_count = count_i_tags(line.strip())
            
            # Check if line has 2 or more of either tag
            if opening_count >= 2 or closing_count >= 2:
                multiple_tag_lines.append(line.strip())
    
    # Write output file
    with open('check.txt', 'w', encoding='utf-8') as file:
        for line in multiple_tag_lines:
            file.write(line + '\n')
            
    print(f"処理が完了しました。{len(multiple_tag_lines)}行が check.txt に出力されました。")
    
except FileNotFoundError:
    print("入力ファイル 'text' が見つかりません。")
except Exception as e:
    print(f"エラーが発生しました: {str(e)}")