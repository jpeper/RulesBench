from autogen import UserProxyAgent, Agent
from autogen import GroupChat, GroupChatManager
from autogen.agentchat.contrib.agent_builder import AgentBuilder
import autogen
import sys
import os
import time
from langchain_community.chat_models import AzureChatOpenAI
from langchain_community.llms import Replicate
import time
import asyncio
import os
import hashlib
import html
import pdb
from langchain.schema import HumanMessage
from dotenv import load_dotenv

from tqdm import tqdm

import asyncio
import logging
from langchain.schema import HumanMessage

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

load_dotenv('config/.env')

class RateLimiter:
    """
    A simple rate limiter that allows up to max_calls in a given period (in seconds).
    """
    def __init__(self, max_calls, period):
        self.max_calls = max_calls
        self.period = period
        self.calls = []  # timestamps of recent calls

    async def acquire(self):
        now = time.monotonic()
        # Remove calls older than the period
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            # Wait until the oldest call is outside the period window
            sleep_time = self.period - (now - self.calls[0])
            await asyncio.sleep(sleep_time)
            # After sleeping, update the timestamps
            now = time.monotonic()
            self.calls = [t for t in self.calls if now - t < self.period]
        self.calls.append(time.monotonic())

async def invoke_async_nonbatched(llm, prompts, name, concurrency_limit=25):
    """
    Sends each prompt as an individual asynchronous call by calling the synchronous llm.invoke method.
    Limits the number of concurrent calls using a semaphore, rate-limits calls to 250 per minute,
    and displays a progress bar and timer.

    Args:
        llm: The LLM instance.
        prompts (list): A list of prompts.
        name (str): Name of the model.
        concurrency_limit (int): Maximum number of concurrent calls.

    Returns:
        list: A list of responses.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    # Rate limiter: 250 calls per 60 seconds
    if 'LLaMA' in name:
        rate_limiter = RateLimiter(max_calls=500, period=60)
        semaphore = asyncio.Semaphore(1)
    elif 'Claude' in name:
        rate_limiter = RateLimiter(max_calls=1000, period=30)
        semaphore = asyncio.Semaphore(2)
    else:
        rate_limiter = RateLimiter(max_calls=10000, period=60)
    pbar = tqdm(total=len(prompts), desc="Processing prompts", unit="prompt")
    start_time = time.perf_counter()

    async def single_call(prompt):
        async with semaphore:
            await rate_limiter.acquire()
            try:
                # Run the synchronous llm.invoke in a separate thread.
                response = await asyncio.to_thread(llm.invoke, prompt)
                if "llama" not in name.lower():
                    return response.content
                return response
            except Exception as e:
                if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                    tqdm.write(f"Content filter triggered for prompt: {prompt}")
                    return "No Response"
                else:
                    tqdm.write(f"Error processing prompt: {prompt}\n{e}")
                    return "No Response"
            finally:
                pbar.update(1)

    tasks = [asyncio.create_task(single_call(prompt)) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - start_time
    pbar.close()
    tqdm.write(f"Processed {len(prompts)} prompts in {elapsed:0.2f} seconds.")
    return results


# Configure logging
def llm_infer(ex, model='gpt-4o', use_json=False):
    """
    Input: the llm prompt (string)
    Output: llm response (string?)
    Performs error-handling for batched async runs -- this is needed due to the Azure content filter
    """
    if type(ex) == str:
        ex = [ex]
        
    if "llama" in model:
        llm = Replicate(model=model, model_kwargs={"temperature": 0.0, "max_length": 4096, "top_p": 1})

    else:
        model_kwargs = {}
        if use_json == True:
            model_kwargs = {"response_format": {"type": "json_object"}}


        llm = AzureChatOpenAI(
            openai_api_version="2024-02-15-preview",
            azure_deployment=model,
            temperature=0.0,
            max_tokens = 8192,
            max_retries=7,
            model_kwargs=model_kwargs
        )

    async_nonbatched = True
    if async_nonbatched:
        # Use the new async non-batched variant that calls llm.invoke.
        return asyncio.run(invoke_async_nonbatched(llm, ex, model))
    
    async def invoke_concurrently_batch(llm, exs):
        batch_size = 10  # Adjust as needed
        results = []
        for i in range(0, len(exs), batch_size):
            batch = exs[i:i+batch_size]
            print("inferring")
            try:
                # Attempt to process the entire batch]
                resp = await llm.abatch(
                    [[HumanMessage(content=query)] for query in batch],
                    {'max_concurrency': 10}
                )
                # Append the responses to the results
                results.extend([r for r in resp])
            except Exception as e:
                if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                    print(f"Batch starting at index {i} triggered content filter. Retrying individually.")
                # If batch fails, process each query individually
                for idx, query in enumerate(batch):
                    try:
                        resp = await llm.abatch(
                            [[HumanMessage(content=query)]],
                            {'max_concurrency': 1}
                        )
                        results.append(resp[0])
                    except Exception as e:
                        if 'content filter being triggered' in str(e) or 'was filtered due to' in str(e):
                            print(f"Query at index {i+idx} triggered content filter.")
                            # Return "No Response" for content-filtered queries
                            results.append("No Response")
                        else:
                            print(e)
                            results.append("No Response")
    
        return results

    s = time.perf_counter()

    num_examples = 1
    if isinstance(ex, list):
        output = asyncio.run(invoke_concurrently_batch(llm, ex))
        num_examples = len(ex)
    else:
        output = asyncio.run(invoke_concurrently_batch(llm, [ex]))
    elapsed = time.perf_counter() - s
    print("\033[1m" + f"Concurrently executed {num_examples} examples in {elapsed:0.2f} seconds." + "\033[0m")

    if isinstance(ex, list):
        return [elt.content for elt in output]
    else:
        return output[0].content