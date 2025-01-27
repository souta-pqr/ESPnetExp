import re

with open('text', 'r') as file:
    lines = file.readlines()

changed_lines = []
for line in lines:
    changed_line = re.sub(r'\(([^)]+)\s+F\)', r'\1', line)
    changed_lines.append(changed_line)

with open('changed_text', 'w') as file:
    file.writelines(changed_lines)