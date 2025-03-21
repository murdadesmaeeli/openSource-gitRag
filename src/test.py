import os
from pathlib import Path
# Get the current file's directory and navigate to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
from dotenv import load_dotenv
load_dotenv()
import os
import json
import asyncio
import nest_asyncio
import logging
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Initialize OpenAI async client
client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'),base_url=os.getenv('BASE_URL'))

# Token Rate Limit Configuration
# token_limit_per_minute = 2000000
token_limit_per_minute = 2000000
char_limit=200000
token_counter = []
token_counter_time_window = 60  # seconds

# Chunk text by characters
def chunk_text_by_characters(text, char_limit=char_limit):
    for i in range(0, len(text), char_limit):
        yield text[i:i + char_limit]

# Rate limiting enforcement
async def enforce_rate_limit():
    current_time = time.time()
    while token_counter and token_counter[0] < current_time - token_counter_time_window:
        token_counter.pop(0)

    if len(token_counter) >= token_limit_per_minute:
        earliest_token_time = token_counter[0]
        sleep_time = (earliest_token_time + token_counter_time_window) - current_time
        if sleep_time > 0:
            logging.info(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)

# GPT-4 API Call
async def call_gpt4(question, text_chunk, max_retries=5):
    prompt = f"""
    You are going to look at the file contents in separate chunks based on the chunks that are returned. 
    Return a JSON with key \"response\" containing a list of file paths starting with 'srcRepo'.

    Goal/Question: {question}

    Context:
    {text_chunk}
    """

    retries = 0
    while retries <= max_retries:
        await enforce_rate_limit()
        try:
            response = await client.chat.completions.create(
                # model="gpt-4o",
                model="us.amazon.nova-micro-v1:0",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                top_p=0.95,
                response_format={"type": "json_object"}
            )

            usage = response.usage
            total_tokens_used = usage.total_tokens
            token_counter.extend([time.time()] * total_tokens_used)
            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logging.error(f"Unexpected error: {e}")

        retries += 1
        sleep_time = 2 ** retries
        logging.warning(f"Retrying after {sleep_time} seconds...")
        await asyncio.sleep(sleep_time)

    logging.error("Max retries exceeded.")
    return None

# Process chunks asynchronously with concurrency control
async def process_text_chunks(question, text_chunks, max_concurrent_requests=(token_limit_per_minute//char_limit)-1):
    semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def sem_task(chunk):
        async with semaphore:
            return await call_gpt4(question, chunk)

    tasks = [sem_task(chunk) for chunk in text_chunks]
    return await asyncio.gather(*tasks)

# Main execution
async def main(question):
    input_file = 'tmp/file_tree.txt'
    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()

        text_chunks = list(chunk_text_by_characters(text, char_limit=128000))
        results = await process_text_chunks(question, text_chunks)

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)

        logging.info("Processing complete. Results saved to 'results.json'.")

    except FileNotFoundError:
        logging.error(f"Input file '{input_file}' not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Apply nest_asyncio for environments like Jupyter
nest_asyncio.apply()

# Example usage
user_question = "how to get current month cloud costs with boto3 sdk?"
asyncio.run(main(user_question))

