def update_file(input_file, output_file):
    with open(input_file, 'r') as f, open(output_file, 'w') as out:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            id_part, _ = parts
            id_parts = id_part.split('_')
            if len(id_parts) != 4:
                continue
            new_start = round(float(id_parts[2]) / 1000, 3)
            new_end = round(float(id_parts[3]) / 1000, 3)
            new_id_part = '_'.join([id_parts[0], id_parts[1]])
            out.write(f"{id_part} {new_id_part} {new_start:.3f} {new_end:.3f}\n")

# ファイルを更新する
update_file('utt2spk', 'segments')
