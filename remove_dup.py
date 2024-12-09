def remove_duplicates(input_file, output_file):
    """
    Removes duplicate lines from a text file and saves unique lines to a new file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output text file with unique lines.
    """
    try:
        # Create a set to store unique lines
        unique_lines = set()
        org_count = 0
        uni_count = 0

        # Read the input file and collect unique lines
        with open(input_file, 'r') as infile:
            for line in infile:
                stripped_line = line.strip()  # Remove leading/trailing whitespaces
                org_count += 1
                if stripped_line not in unique_lines:
                    uni_count += 1
                    unique_lines.add(stripped_line)

        print(org_count, uni_count)

        # Write the unique lines to the output file
        with open(output_file, 'w') as outfile:
            for line in sorted(unique_lines):  # Sorting optional, for consistent output
                outfile.write(line + '\n')

        print(f"Duplicates removed. Unique lines saved to '{output_file}'.")
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
input_file_path = "successful_tickers.txt"  # Replace with your input file path
output_file_path = 'successful_tickers.txt'  # Replace with your desired output file path
input_file_path1 = "unsuccessful_tickers.txt"  # Replace with your input file path
output_file_path1 = 'unsuccessful_tickers.txt'  # Replace with your desired output file path
remove_duplicates(input_file_path, output_file_path)
remove_duplicates(input_file_path1, output_file_path1)