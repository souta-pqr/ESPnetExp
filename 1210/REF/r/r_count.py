def find_lines_with_multiple_r_tags(file_path):
    try:
        # Open and read the file with UTF-8 encoding
        with open(file_path, 'r', encoding='utf-8') as file:
            # Process the file line by line
            # Process each line
            for line in file:
                # Count the number of (R) tags in the line
                r_count = line.count('(R)')
                
                # If there are 2 or more (R) tags, print the line
                if r_count >= 2:
                    print(line)
                    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except UnicodeDecodeError:
        print(f"Error: Unable to read the file. Please check the file encoding.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

# File path
file_path = 'text'  # ファイル名を指定

# Run the function
find_lines_with_multiple_r_tags(file_path)