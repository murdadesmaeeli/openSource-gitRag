import os
import mimetypes
import boto3
import json
from botocore.exceptions import ClientError



bedrock_client = boto3.client('bedrock-runtime', region_name='us-west-2')

model_id = "us.meta.llama3-2-3b-instruct-v1:0"





def create_file_tree_with_contents(directory, output_file='file_tree.txt'):
    def summarize(file_content):
        # Invoke the model to summarize the code provided into a smaller snippet
        prompt = """## Instruction
Your task is to summarize the given code in the <code> </code> tags into a concise description that captures all the key features and functionality, while omitting any unnecessary or redundant information.

To produce an effective summary, please follow these guidelines:

### Code Summarization Guidelines
- Read through the code carefully to understand its purpose, logic flow, and implementation details.
- Identify the core algorithms, data structures, and programming concepts used in the code.
- Determine the main features and functionalities provided by the code.
- Describe the code's inputs, outputs, and any important parameters or configurations.
- Explain any key design patterns, optimization techniques, or performance considerations implemented.
- Use clear and concise language to convey the essential information about the code's behavior and capabilities.
- Avoid including minor implementation details, comments, or code snippets unless they are crucial for understanding the overall functionality.

### Code to Summarize
<code>""" + file_content +  """</code>"""
        # Embed the prompt in Llama 3's instruction format.
        formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>       
        """
        # Format the request payload using the model's native structure.
        native_request = {
            "prompt": formatted_prompt,
            "max_gen_len": 512,
            "temperature": 0.5,
        }
        # Convert the native request to JSON.
        request = json.dumps(native_request)
        try:
            # Invoke the model with the request.
            response = bedrock_client.invoke_model(modelId=model_id, body=request)
            # Decode the response body.
            model_response = json.loads(response["body"].read())
            # Extract and print the response text.
            response_text = model_response["generation"]
            return response_text
        except (ClientError, Exception) as e:
            return f"ERROR: Can't invoke '{model_id}'. Reason: {e}"
        exit(1)
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
                            summary=summarize(contents)
                            # Write the contents with proper indentation
                            for line in summary.splitlines():
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
create_file_tree_with_contents('srcRepo')
