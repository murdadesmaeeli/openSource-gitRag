import os
import os
from pathlib import Path
# Get the current file's directory and navigate to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
from dotenv import load_dotenv
load_dotenv()
import mimetypes





def create_file_tree_with_contents(directory, output_file='tmp/file_tree.txt'):
    def is_text_file(file_path):
        text_extensions = {
            '.txt', '.md', '.py', '.js', '.java', '.c', '.cpp', '.h', '.css',
            '.html', '.xml', '.json', '.yaml', '.yml', '.ini', '.conf', '.sh',
            '.bat', '.csv', '.log'
        }
        
        if os.path.splitext(file_path)[1].lower() in text_extensions:
            return True
            
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type and mime_type.startswith('text'):
            return True
            
        try:
            with open(file_path, 'rb') as f:
                chunk = f.read(1024)
                return not bool(b'\x00' in chunk)
        except Exception:
            return False

    def write_tree(out_file, root, prefix=""):
        items = os.listdir(root)
        for index, item in enumerate(sorted(items)):
            item_path = os.path.join(root, item)
            is_last = index == len(items) - 1

            # Write the file/directory name
            connector = "└── " if is_last else "├── "
            out_file.write(f"{prefix}{connector}{item}\n")

            if os.path.isdir(item_path):
                new_prefix = prefix + ("    " if is_last else "│   ")
                write_tree(out_file, item_path, new_prefix)
            else:
                if is_text_file(item_path):
                    try:
                        # Add separator before file contents
                        out_file.write(f"{prefix}{'    ' if is_last else '│   '}-Contents-\n")
                        
                        with open(item_path, 'r', encoding='utf-8', errors='ignore') as f:
                            contents = f.read()
                            # Write the contents with proper indentation
                            for line in contents.splitlines():
                                out_file.write(f"{prefix}{'    ' if is_last else '│   '}    {line}\n")
                         
                        # Add separator after file contents
                        out_file.write(f"{prefix}{'    ' if is_last else '│   '}--\n")
                    except Exception as e:
                        out_file.write(f"{prefix}{'    ' if is_last else '│   '}[Error reading file: {e}]\n")
                else:
                    out_file.write(f"{prefix}{'    ' if is_last else '│   '}[Skipped non-text file]\n")

    with open(output_file, 'w', encoding='utf-8') as out_file:
        out_file.write(f"{os.path.basename(directory)}/\n")
        write_tree(out_file, directory)

# Usage example:
create_file_tree_with_contents('gitRagRepo')
