import os
import mimetypes

def create_file_tree_with_contents(directory, output_file='file_tree.txt'):
    def is_text_file(file_path):
        # Guess the mime type of the file
        mime_type, _ = mimetypes.guess_type(file_path)
        # Consider text files and those with no mime type as readable
        return mime_type is None or mime_type.startswith('text')

    def write_tree(out_file, root, prefix=""):
        items = os.listdir(root)
        for index, item in enumerate(sorted(items)):
            item_path = os.path.join(root, item)
            is_last = index == len(items) - 1

            # Determine the prefix for this item
            connector = "└── " if is_last else "├── "
            out_file.write(f"{prefix}{connector}{item}\n")

            # If item is a directory, recurse into it
            if os.path.isdir(item_path):
                new_prefix = prefix + ("    " if is_last else "│   ")
                write_tree(out_file, item_path, new_prefix)
            else:
                # If item is a file, check if it is a text file
                if is_text_file(item_path):
                    try:
                        with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                            contents = f.readlines()
                            file_prefix = prefix + ("    " if is_last else "│   ")
                            for line in contents:
                                out_file.write(f"{file_prefix}    {line}")
                    except Exception as e:
                        # Handle cases where the file can't be read (e.g., encoding issues)
                        out_file.write(f"{prefix}    [Error reading file: {e}]\n")
                else:
                    # Skip non-text files
                    out_file.write(f"{prefix}    [Skipped non-text file]\n")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"{os.path.basename(directory)}/\n")
        write_tree(out_file, directory)

# Usage example:
create_file_tree_with_contents('srcRepo')
