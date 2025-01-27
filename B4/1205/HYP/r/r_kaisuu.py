def analyze_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    r_count = 0
    multiple_r_lines = []
    
    for line in lines:
        r_count_in_line = line.count('(R')
        if r_count_in_line > 0:
            r_count += 1
            if r_count_in_line > 1:
                multiple_r_lines.append(line)
    
    return r_count, multiple_r_lines

def main():
    filename = 'text'  # 入力ファイル名
    total_r_count, multiple_r_lines = analyze_text(filename)
    
    print(f"Total number of lines with (R: {total_r_count}")
    print("\nLines with multiple (R occurrences:")
    for line in multiple_r_lines:
        print(line.strip())

if __name__ == "__main__":
    main()