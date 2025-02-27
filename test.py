import os
import json
import aiohttp
import asyncio
import pandas as pd
import nest_asyncio
import logging
import time
 
 
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Token Rate Limit Configuration
token_limit_per_minute = 2000000
token_counter = []
token_counter_time_window = 60  # seconds

# Function to chunk text into segments of 42,000 words
def chunk_text(text, word_limit=25000):
    words = text.split()
    for i in range(0, len(words), word_limit):
        yield ' '.join(words[i:i + word_limit])

# Function to enforce rate limiting
async def enforce_rate_limit():
    current_time = time.time()
    # Remove tokens older than 60 seconds
    while token_counter and token_counter[0] < current_time - token_counter_time_window:
        token_counter.pop(0)
    # Calculate total tokens in the last minute
    if len(token_counter) >= token_limit_per_minute:
        earliest_token_time = token_counter[0]
        sleep_time = (earliest_token_time + token_counter_time_window) - current_time
        if sleep_time > 0:
            logging.info(f"Rate limit approaching. Sleeping for {sleep_time:.2f} seconds.")
            await asyncio.sleep(sleep_time)

# Asynchronous function to call the GPT-4 API
async def call_gpt4(session, question, text_chunk,max_retries=5):
    # Construct the prompt with the user-provided question
    prompt = f"""
    You are going to look at the file contents in separate big chunks 
    based on the chunks that are returned that files that might need change based on the goal/question that 
    you get. Return a JSON with key "response" and the value is a list of strings that are full relative paths
    to the files being referenced. the path should start with srcRepo

    Goal/Question: {question}

    Context:
    {text_chunk}
    """
    
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0,
        "top_p": 0.95,
        "response_format": {"type":"json_object"}
    }
    retries = 0
    while retries <= max_retries:
        await enforce_rate_limit()  # Check the rate limit before making the API call
        
        try:
            async with session.post(ENDPOINT, headers=headers, json=payload) as response:
                if response.status == 429:
                    retries += 1
                    retry_after = response.headers.get("Retry-After")
                    sleep_time = float(retry_after) if retry_after else 2 ** retries
                    logging.warning(f"Received 429 Too Many Requests. Retrying after {sleep_time} seconds...")
                    await asyncio.sleep(sleep_time)
                    continue
                
                response_json = await response.json()
                logging.debug(json.dumps(response_json, indent=2))
                
                if 'choices' in response_json:
                    # Update token usage
                    usage = response_json.get('usage', {})
                    total_tokens_used = usage.get('total_tokens', 0)
                    token_counter.extend([time.time()] * total_tokens_used)
                    
                    return json.loads(response_json['choices'][0]['message']['content'])
                else:
                    logging.error("Unexpected response format: 'choices' key missing.")
                    logging.error(f"Response content: {response_json}")
                    return None
                
        except aiohttp.ClientError as e:
            logging.error(f"HTTP error occurred: {e}")
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
        
        retries += 1

    logging.error("Max retries exceeded.")
    return None

# Asynchronous function to process all text chunks
async def process_text_chunks(question, text_chunks):
    async with aiohttp.ClientSession() as session:
        tasks = [call_gpt4(session, question, chunk) for chunk in text_chunks]
        return await asyncio.gather(*tasks)

# Main function to read file, chunk it, and process each chunk
async def main(question):
    try:
        # Read your input text file
        with open('cleaned_file_tree.txt', 'r',encoding='utf-8') as file:
            text = file.read()
        
        # Chunk the text
        text_chunks = list(chunk_text(text))
        
        # Process each chunk asynchronously
        results = await process_text_chunks(question, text_chunks)
        
        # Save the results to a JSON file
        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)
        logging.info("Processing complete. Results saved to 'results.json'.")
    except FileNotFoundError:
        logging.error("Input file 'file_tree.txt' not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Apply nest_asyncio to handle nested async calls in some environments
nest_asyncio.apply()

# Example question from the user
user_question = "Which files need updates based on the new security guidelines?"

# Run the main function with the user's question
asyncio.run(main(user_question))
