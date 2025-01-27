def count_f_marks(line):
    """Count occurrences of (F 文字) pattern in a line"""
    return line.count('(D ')

# Read the input file and process each line
with open('text', 'r', encoding='utf-8') as input_file:
    # Store lines with 2 or more (F ) marks
    lines_with_multiple_f = []
    
    for line in input_file:
        if count_f_marks(line) >= 2:
            lines_with_multiple_f.append(line)

# Write the filtered lines to over.txt
with open('over.txt', 'w', encoding='utf-8') as output_file:
    output_file.writelines(lines_with_multiple_f)