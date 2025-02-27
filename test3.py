from dotenv import load_dotenv
import os
import fnmatch
import aiofiles
import asyncio

from typing import List, Optional, Dict, TypedDict, Union, Set
from pydantic import BaseModel, Field
from langchain_text_splitters import RecursiveCharacterTextSplitter
import gradio as gr
import aiohttp
import json

load_dotenv(override=True)

try:
    AZURE_CHEAP_MODEL_API_KEY = os.getenv('AZURE_CHEAP_MODEL_API_KEY')
    AZURE_CHEAP_MODEL_API_ENDPOINT = os.getenv('AZURE_CHEAP_MODEL_API_ENDPOINT')
    AZURE_CHEAP_MODEL_API_VERSION= os.getenv('AZURE_CHEAP_MODEL_API_VERSION')
except Exception as e:
    print("Couldn't get API keys.")
    print(e)


async def prompt_cheap_gpt_model(json_schema, system_message, human_message):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_CHEAP_MODEL_API_KEY,
    }

    formatted_prompt = {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": human_message},
        ],
        "temperature": 0,
        "top_p": 0.95,
        "response_format": {"type": "json_schema", "json_schema": json_schema}
    }

    async with aiohttp.ClientSession() as session:
        retry_attempts = 6
        for retry_counter in range(retry_attempts):
            try:
                async with session.post(AZURE_CHEAP_MODEL_API_ENDPOINT, json=formatted_prompt, headers=headers) as response:
                    if response.status == 429:
                        # Handle rate limiting by waiting before retrying
                        retry_after = int(response.headers.get("Retry-After", 2**retry_counter))
                        print(f"Rate limited. Retrying in {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        continue
                    response_json = await response.json()
                    
                    try:
                        llm_output = json.loads(response_json['choices'][0]['message']['content'])
                        return llm_output.get('properties', llm_output)
                    except Exception as e:
                        print(f"Error occurred: {e}. Response content: {response_json}")
                        return None
                    
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue


class UserQueryCodeMetadata(BaseModel):
    """Extracted metadata about code files from text."""
    # Each field is an `optional` -- this allows the model to decline to extract it!
    name_of_files: Optional[List[str]] = Field(..., description="Specific filenames with extensions mentioned in the user question.")
    extensions: Optional[List[str]] = Field(..., description="A list of code file extensions mentioned in the user question.")
    files_to_ignore: List[str] = Field(description="A list of code files and directories to be excluded from consideration based on the user question.", default_factory=lambda: ['*.pack', '*.bin', '*.jpeg', '*.png', '.git/', '*.json', '*.ico', '*.map'])


async def filter_files_from_user_prompt(user_query: str):
    system_message = (
        """
        # Instruction:
        You are an expert at extracting metadata of code files from the text. Only extract relevant information from the text.
        If you do not know the value of an attribute asked to extract, return null for the attribute's value.

        # Output format:
        Produce the output in JSON format according to the provided schema. 

        # Examples:
            - Example 1:
            Input: "I have implemented a custom cv model in model.py. The model is called from the main python file and deployed with gradio. However, I get the following error when trying to run it: Error: app.js file not found. Ignore any text or yaml files"
            Output: {
                "name_of_files": ['model.py', 'main.py', 'app.js'],
                "extensions": ['.py', '.js'],
                "files_to_ignore": ['*.yaml', '*.txt', '*.pack', '*.bin', '*.jpeg', '*.png', '.git/', '*.json', '*.ico', '*.map']
            }
            - Example 2: 
            Input: "Which java file is responsible for handling network requests? Do not include any javascript files even if they may handle requests partially. Ignore the html config for now and all files in myImages folder."
            Output: {
                "name_of_files": null,
                "extensions": ['.java'],
                "files_to_ignore": ['*.js', 'config.html', 'myImages/', '*.pack', '*.bin', '*.jpeg', '*.png', '.git/', '*.json', '*.ico', '*.map']
            }
        """
    )
    
    llm_response = await prompt_cheap_gpt_model(json_schema={"name": "code_files_metadata", "schema": UserQueryCodeMetadata.model_json_schema()}, system_message=system_message, human_message=user_query) 
    return llm_response


def should_include_file(file, path, specific_files=None, extensions=None, ignore_patterns=None):
    if ignore_patterns and (file in ignore_patterns or any(fnmatch.fnmatch(path, pattern) for pattern in ignore_patterns)):
        return False
    if extensions and not any(path.endswith(ext) for ext in extensions):
        return False
    if specific_files and file not in specific_files:
        return False
    return True

async def lookup_directory_files(directory, specific_files=None, extensions=None, ignore_patterns=None, is_indent=False):
    if not os.path.isdir(directory):
        return [os.path.normpath(directory).replace(os.sep, '/')]

    src_files = []
    for root, dirs, files in await asyncio.to_thread(os.walk, directory):
        dirs[:] = [d for d in dirs if should_include_file(None, f'{d}/', ignore_patterns=ignore_patterns)]

        if is_indent:
            normalized_root = os.path.normpath(root).replace(os.sep, '/')
            level = normalized_root.replace(directory, '').count(os.sep)
            indented_root = f"{'│   ' * level}├── {normalized_root}/"
            src_files.append(indented_root)

        for file in files:
            full_path = os.path.join(root, file)
            normalized_path = os.path.normpath(full_path).replace(os.sep, '/')
            if should_include_file(file, normalized_path, specific_files, extensions, ignore_patterns):
                if is_indent:
                    indented_path = f"{'│   ' * (level + 1)}├── {normalized_path}"
                    src_files.append(indented_path)
                else:
                    src_files.append(normalized_path)

    return src_files


'''Parse filtered files and their contents into a dictionary
    - If the user mentions a specific file in their prompt, only that file will be parsed.
    - If the user mentions file type, only files with that extension will be parsed.
    - If the user specifies any files to ignore, they will be skipped.
'''

async def get_directory_files_contents(files):
    files_and_content = {}
    for file in files:
        try:
            async with aiofiles.open(file, encoding='utf-8', errors='ignore') as f:
                files_and_content[file] = await f.read()
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    return files_and_content

async def get_files(directory, files_to_lookup, extensions_to_lookup, files_to_ignore):
    file_paths = await lookup_directory_files(directory, specific_files=files_to_lookup, extensions=extensions_to_lookup, ignore_patterns=files_to_ignore, is_indent=False)
    file_contents = await get_directory_files_contents(file_paths)
    return file_contents


def split_codefile_to_smaller_chunks(code_file_content, prefered_chunk_size=300000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=prefered_chunk_size, separators=['\n\n\n', '\n\n', '\n', ','], chunk_overlap=0, length_function=len, is_separator_regex=False)
    documents = text_splitter.create_documents([code_file_content])
    new_chunks = [chunk.page_content for chunk in documents]
    return new_chunks

def chunk_documents(repo_contents: Dict[str, str], max_chunk_size=300000) -> List[str]:
    all_chunks = []
    current_chunk = []
    current_chunk_len = 0

    for filepath, file_content in repo_contents.items():
        try:
            full_content = f"{filepath}: {file_content}"
            curr_file_len = len(full_content)

            # Attempt to split the file into smaller chunks if it's too large
            if curr_file_len > max_chunk_size:
                try:
                    smaller_chunks = split_codefile_to_smaller_chunks(full_content, max_chunk_size)
                    current_file_pieces = [f"{filepath}: {chunk}" for chunk in smaller_chunks]
                except Exception as e:
                    print(f"Error splitting file {filepath} into smaller chunks: {e}")
                    current_file_pieces = [full_content]
            else:
                current_file_pieces = [full_content]

            for code_subchunk in current_file_pieces:
                subchunk_len = len(code_subchunk)
                # If adding this subchunk exceeds the max size, push the current chunk to all chunks
                if current_chunk_len + subchunk_len > max_chunk_size:
                    all_chunks.append("\n\n\n\n\n".join(current_chunk))
                    current_chunk = []
                    current_chunk_len = 0

                current_chunk.append(code_subchunk)
                current_chunk_len += subchunk_len

        except Exception as e:
            print(f"Error processing file {filepath}: {e}")

    # Add any remaining content in the current chunk
    if current_chunk:
        all_chunks.append("\n\n\n\n\n".join(current_chunk))
    
    return all_chunks


'''Rank the relevance of docs to user query using a cheap GPT model'''

class RankDocRelevanceToUserQuery(BaseModel):
    """Scored code files based on their relevancy to the user question"""
    file: str = Field(..., description=" An absolute path of the code file.")
    summary_of_code_file: str = Field(..., description=" A concise summary of the main features and functionality of the code file.")
    score: float = Field(..., description="A relevance score of how well the code file matches the user's search intents.")
    
class ScoredFiles(BaseModel):
    """Extracted data about code files."""
    scored_files: Set[RankDocRelevanceToUserQuery]

async def rank_doc(user_query: str, code_file_contents: str, repo_overview: str):
    system_message = f"""You are an expert at scoring the relevance and strength of code files in answering a user's question. Your task is to extract and summarize key information from each code file to determine how effectively it addresses the user's question.

            # Scoring Guidelines
            1. **Relevance**: Assign scores based on how directly a code file contributes to answering the user's question. The most relevant file should be scored as 1.0.
            2. **Relative Strength**: Evaluate the significance of each file in the context of the entire repository. Adjust scores to reflect both the individual file's relevance and its importance within the repository.
            3. **Irrelevant Files**: Assign a score of 0 to files that do not meet any of the user's search intents.

            # Important
            - Prioritize files that directly address the user's question with clear and actionable information.
            - Contextualize each file's contribution within the broader scope of the repository: {repo_overview}.
            - Ensure your scoring reflects both the relevance to the user query and the relative strength of the file.

            # Output format:
            Produce the output in JSON format according to the provided schema.

            # Examples:
            - Example 1:
            Input: "Which python files should I modify in my repo to improve chunking strategy? Ignore all yml files. \n\n Example Code file: test_repomap/scripts/chunking.py import argparse import dataclasses import json import os  from azure.identity import DefaultAzureCredential from azure.core.credentials import AzureKeyCredential from azure.keyvault.secrets import SecretClient from azure.ai.formrecognizer import DocumentAnalysisClient  from data_utils import chunk_directory  def get_document_intelligence_client(config, secret_client):     print(Setting up Document Intelligence client...)     secret_name = config.get(document_intelligence_secret_name)      if not secret_client or not secret_name:         print(No keyvault url or secret name provided in config file. Document Intelligence client will not be set up.)         return None      endpoint = config.get(document_intelligence_endpoint)     if not endpoint:         print(No endpoint provided in config file. Document Intelligence client will not be set up.)         return None          try:         document_intelligence_secret = secret_client.get_secret(secret_name)         os.environ[FORM_RECOGNIZER_ENDPOINT] = endpoint         os.environ[FORM_RECOGNIZER_KEY] = document_intelligence_secret.value          document_intelligence_credential = AzureKeyCredential(document_intelligence_secret.value)          document_intelligence_client = DocumentAnalysisClient(endpoint, document_intelligence_credential)         print(Document Intelligence client set up.)         return document_intelligence_client     except Exception as e:         print(Error setting up Document Intelligence client: .format(e))         return None   if __name__ == __main__:     parser = argparse.ArgumentParser()     parser.add_argument(--input_data_path, type=str, required=True)     parser.add_argument(--output_file_path, type=str, required=True)     parser.add_argument(--config_file, type=str, required=True)      args = parser.parse_args()      with open(args.config_file) as f:         config = json.load(f)      credential = DefaultAzureCredential()      if type(config) is not list:         config = [config]          for index_config in config:         # Keyvault Secret Client         keyvault_url = index_config.get(keyvault_url)         if not keyvault_url:             print(No keyvault url provided in config file. Secret client will not be set up.)             secret_client = None         else:             secret_client = SecretClient(keyvault_url, credential)          # Optional client for cracking documents         document_intelligence_client = get_document_intelligence_client(index_config, secret_client)          # Crack and chunk documents         print(Cracking and chunking documents...)          chunking_result = chunk_directory(                             directory_path=args.input_data_path,                              num_tokens=index_config.get(chunk_size, 1024),                             token_overlap=index_config.get(token_overlap, 128),                             form_recognizer_client=document_intelligence_client,                             use_layout=index_config.get(use_layout, False),                             njobs=1)                  print(fProcessed chunking_result.total_files files)         print(fUnsupported formats: chunking_result.num_unsupported_format_files files)         print(fFiles with errors: chunking_result.num_files_with_errors files)         print(fFound len(chunking_result.chunks) chunks)          print(Writing chunking result to ....format(args.output_file_path))         with open(args.output_file_path, w) as f:             for chunk in chunking_result.chunks:                 id = 0                 d = dataclasses.asdict(chunk)                 # add id to documents                 d.update(id: str(id))                 f.write(json.dumps(d) + \n)                 id += 1         print(Chunking result written to ..format(args.output_file_path))"
            Output: {{
                "scoredFiles": [
                    {{
                    "file": "example_repo/example_scripts/chunking.py",
                    "summary_of_code_file": "The code file sets up a Document Intelligence client with Azure secrets, processes and chunks documents from an input directory and saves the results to an output file.",
                    "score": 0.85
                    }},
                    {{
                    "file": "example_repo/feature_request.md",
                    "summary_of_code_file": "The code file is a readme file that summarizes how to make a request to add additional features",
                    "score": 0.0
                    }}
                ]
            }}
            """

    user_query_with_codefile = f'User Question: {user_query} \n\n Code file: {code_file_contents}'
    llm_response = await prompt_cheap_gpt_model(json_schema={"name": "scored_files_information", "schema": ScoredFiles.model_json_schema()}, system_message=system_message, human_message=user_query_with_codefile)
    return llm_response


# Helper function for printing docs
def prety_print_summaries_and_scores(summaries_and_scores):
    print(
        f"\n{'-' * 100}\n".join(
            [f"Document {summary_entry['file']}: Score: {summary_entry['score']} \n Summary: {summary_entry['summary_of_code_file']}" for summary_entry in summaries_and_scores]
        )
    )

def flatten_scored_files(scored_files):
    flat_list = []
    if scored_files:
        for item in scored_files:
            flat_list.extend(item.get('scored_files', ''))
    return flat_list

async def select_top_k_documents(user_query, code_chunks, score_threshold=0.7, repo_overview="", print_debugging=False):
    try:
        scored_files = await asyncio.gather(*[rank_doc(user_query, code_chunk, repo_overview) for code_chunk in code_chunks])
    except Exception as e:
        print(f"Error during document ranking: {e}")
        return []

    try:
        flattened_scored_files = flatten_scored_files(scored_files)
        flattened_scored_files.sort(key=lambda dict_entry: dict_entry.get('score', 0), reverse=True)

        if print_debugging:
            prety_print_summaries_and_scores(flattened_scored_files)

        top_k_filepaths = [dict_entry for dict_entry in flattened_scored_files if dict_entry.get('score', 0) >= score_threshold]

        if print_debugging:
            print(f'Total top k docs selected: {len(top_k_filepaths)}.')

        return top_k_filepaths

    except Exception as e:
        print(f"Error processing flattened scored files: {e}")
        return []
    

'''Generate editing suggestions for most relevant docs using Search-Replace blocks'''

class EditSuggestion(BaseModel):
    """Suggested code changes to one or more code files."""
    file: str = Field(..., description="The file to be updated.")
    changes: Optional[List[Dict[str, List[str]]]] = Field(description="A list of code changes containing the lines of code being modified ('original_code_lines') and 'updated_code_lines' that replace 'original_code_lines'.", default=None)

class AllSuggestions(BaseModel):
    """Generated edit suggestions to address user question."""
    explanation: str = Field(..., description="Short summary of the necessary code changes to be made.")
    edit_suggestions: List[EditSuggestion]


async def suggest_edits(user_query: str, code_file_contents: str):
    system_message = """You are an expert software developer. Analyze the user's question, assess which changes should be made to relevant code files and implement the neccessary changes to address user question.

                    # Steps
                    1. **Analyze the User Question**: Understand what the user is asking and determine whether it requires changes to the code files.
                    2. **Evaluate the Code**: If changes are needed, determine which most relevant code files should be modified to address the user question.
                    3. **Implement the Changes**: Implement the neccessary changes to the most relevant code files to address the user question.

                    # Output Format
                    Produce the output in JSON format according to the provided schema.

                    # Notes
                    - Ensure only necessary code changes are fully implemented. If no changes are needed to fulfill the user's question, set changes to None
                
                    # Examples
                - Example 1:
                        Input: "Example User Question:\nChange factorial() to use math.factorial\n\nExample Code files:\nexample_repo/mathweb/flask/app.py\nfrom flask import Flask\n\ndef factorial(n):\n    \"compute factorial\"\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)\n    return str(factorial(n))\n\ndef do_random_stuff():\n    print(\"Nothing\")"
                        Output:
                        {
                        "explanation": "To make this change we need to modify `example_repo/mathweb/flask/app.py` to: \n1. Import the math package. \n2. Remove the existing factorial() function.\n3. Update get_factorial() to call math.factorial instead.",
                        "edit_suggestions": [
                            {
                            "file": "example_repo/mathweb/flask/app.py",
                            "changes": [
                                {
                                "original_code_lines": ["from flask import Flask"],
                                "updated_code_lines": ["from flask import Flask\n", "import math\n"]
                                },
                                {
                                "original_code_lines": [
                                    "def factorial(n):\n",
                                    "    \"compute factorial\"\n",
                                    "    if n == 0: return 1\n",
                                    "    else: return n * factorial(n-1)\n\n"
                                ],
                                "updated_code_lines": ["def factorial(n):\n", "    return str(math.factorial(n))"]
                                }
                            ]
                            }
                        ]
                        }
                """

    user_query_with_codefile = f'User Question:\n{user_query}\n\nCode files:\n{code_file_contents}'
    llm_response = llm_response = await prompt_cheap_gpt_model(json_schema={"name": "code_edit_suggestions", "schema": AllSuggestions.model_json_schema()}, system_message=system_message, human_message=user_query_with_codefile)
    return llm_response


def suggestions_to_string(explanation, edit_suggestions, print_debugging=False):
    all_changes = [explanation]
    try:
        if edit_suggestions:
            for edit_suggestion in edit_suggestions:
                file_to_edit = edit_suggestion.get("file", None)
                changes = edit_suggestion.get("changes", None)
                
                if file_to_edit:  # Ensure file_to_edit is not None
                    output_parts = [f'File to edit: {file_to_edit}']

                    if changes:
                        for update in changes:
                            original_code_lines = "\n".join(update['original_code_lines'])
                            updated_code_lines = "\n".join(update['updated_code_lines'])
                            output_parts.append(
                                "<---------- ORIGINAL ---------->\n"
                                f"{original_code_lines}\n"
                                "<====================>\n"
                                f"{updated_code_lines}\n"
                                "<---------- NEW ---------->\n"
                            )

                    all_changes.append("\n".join(output_parts))

        final_output = "\n".join(all_changes)
        return final_output

    except Exception as e:
        print(f"An error occurred: {e}")

async def generate_edit_suggestions(user_query, code_files, print_debugging=False):
    response = []
    try:
        edit_suggestions = await asyncio.gather(*[suggest_edits(user_query, code_file) for code_file in code_files])
        response = [suggestions_to_string(code_update.get("explanation", None), code_update.get("edit_suggestions", None), print_debugging) for code_update in edit_suggestions]
        return "\n\n\n".join(response)
    except Exception as e:
        print(f"Error generating edit suggestions: {e}")
        return ""
    

def format_ranked_documents(docs_paths_and_summaries, docs_dict):
    extracted_docs = {}
    if docs_paths_and_summaries:
        for doc_entry in docs_paths_and_summaries:
            filepath, summary = doc_entry.get('file', ''), doc_entry.get('summary_of_code_file', '')
            content = docs_dict.get(filepath)
            if (content):
                extracted_docs[filepath] = f'Code file summary: {summary}, Code file content: {content}'
    return extracted_docs


async def run_git_rag(user_query, source_repository='srcRepo/', print_debugging=False):
    try:
        filtered_files_info = await filter_files_from_user_prompt(user_query)
        if print_debugging:
            print(filtered_files_info)

        target_files = filtered_files_info.get('name_of_files', [])
        file_extensions = filtered_files_info.get('extensions', [])
        excluded_files = filtered_files_info.get('files_to_ignore', [])

        try:
            filtered_repository_files = await get_files(directory=source_repository, files_to_lookup=target_files, extensions_to_lookup=file_extensions, files_to_ignore=excluded_files)
        except Exception as e:
            print(f"Error retrieving files from repository: {e}")
            return ""

        try:
            repository_tree = await lookup_directory_files( source_repository, target_files, file_extensions, excluded_files, is_indent=True)
        except Exception as e:
            print(f"Error looking up directory files: {e}")
            return ""

        try:
            code_chunks = chunk_documents(filtered_repository_files)
            if print_debugging:
                print(f"Total chunks to score: {len(code_chunks)}")
        except Exception as e:
            print(f"Error chunking documents: {e}")
            return ""

        try:
            top_ranked_document_paths_and_summaries = await select_top_k_documents(user_query, code_chunks, repo_overview=repository_tree, print_debugging=print_debugging)
        except Exception as e:
            print(f"Error selecting top documents: {e}")
            return ""

        try:
            formatted_top_ranked_documents = format_ranked_documents(top_ranked_document_paths_and_summaries, filtered_repository_files)
            top_ranked_docs_chunks = chunk_documents(formatted_top_ranked_documents)
            if print_debugging:
                print(f"Number of chunks to edit: {len(top_ranked_docs_chunks)}")
        except Exception as e:
            print(f"Error formatting or chunking top ranked documents: {e}")
            return ""

        try:
            edit_suggestions = await generate_edit_suggestions(user_query, top_ranked_docs_chunks, print_debugging=print_debugging)
            return edit_suggestions
        except Exception as e:
            print(f"Error generating edit suggestions: {e}")
            return ""

    except Exception as e:
        print(f"Error during run_git_rag execution: {e}")
        return ""
    


def predict(question, history):
    return asyncio.run(run_git_rag(question, print_debugging=False))
    
gr.ChatInterface(predict,
    chatbot=gr.Chatbot(height='auto'),
    textbox=gr.Textbox(placeholder="type here"),
    title="Git Rag Chatbot",
    description="test").launch(share=True, server_name="0.0.0.0", server_port=8000)#auth=('cash','123'))
