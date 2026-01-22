#!/usr/bin/env python3
"""
Remove tokens with +F or +D tags from text file.
Example: "え+F ー+F と" -> "と"
"""

def remove_disfluency_tokens(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            line = line.rstrip('\n')
            if not line:
                f_out.write('\n')
                continue
            
            # Split into utterance ID and tokens
            parts = line.split(None, 1)
            if len(parts) < 2:
                f_out.write(line + '\n')
                continue
            
            utt_id = parts[0]
            tokens_str = parts[1]
            
            # Split tokens and filter out those with +F or +D
            tokens = tokens_str.split()
            filtered_tokens = [token for token in tokens 
                             if not ('+F' in token or '+D' in token)]
            
            # Write result
            if filtered_tokens:
                f_out.write(f"{utt_id} {' '.join(filtered_tokens)}\n")
            else:
                f_out.write(f"{utt_id}\n")

if __name__ == '__main__':
    input_file = 'HYP/text'
    output_file = 'HYP/text_no_disfluency'
    
    remove_disfluency_tokens(input_file, output_file)
    print(f"処理完了: {input_file} -> {output_file}")
    
    # Show some examples
    print("\n変更例:")
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(output_file, 'r', encoding='utf-8') as f:
        new_lines = f.readlines()
    
    for i in range(min(5, len(lines))):
        if lines[i] != new_lines[i]:
            print(f"\n元: {lines[i].rstrip()}")
            print(f"新: {new_lines[i].rstrip()}")
