import time
import os
import json
import asyncio
from typing import Iterable, List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from datasets import load_dataset

from openai_client import OpenAIClient
from prompts import format_input

DataSample = Dict[str, Any]  # e.g. {"nums": List[int], "target": int, ...}
Message = List[Dict[str, str]]  # OpenAI chat format: list of {"role": ..., "content": ...}

INPUT_COLUMN = "input"
PROMPT_COLUMN = "prompt"
OUTPUT_COLUMN = "output"

async def run_batch_inference_smmd(client: OpenAIClient,
                                   model: str,
                                   messages: Iterable[Message],
                                   **kwargs) -> Iterable[str]:
    tasks = [
        client.get_response(model, m, **kwargs)
        for m in messages
    ]
    results = await asyncio.gather(*tasks)
    return results

def pickle_response_object(api_response: Any) -> Dict:
    api_response = api_response.output
    if len(api_response) != 2:
        print(f"[WARNING]: found {len(api_response)=}")
    summary_object, asst_object = api_response

    res = {}
    res['summary'] = [x.text for x in summary_object.summary]
    res['text'] = asst_object.content[0].text
    return res

def engine(df: pd.DataFrame,
           client: OpenAIClient,
           model: str,
           output_path: str,
           batch_size: int,
           **api_kwargs):
    if OUTPUT_COLUMN not in df.columns:
        df[OUTPUT_COLUMN] = None

    for batch_start_idx in tqdm(range(0, len(df), batch_size), desc="Generating responses for inputs"):
        batch_df = df.iloc[batch_start_idx: batch_start_idx + batch_size]
        # batch_prompts = [json.loads(x) for x in batch_df[PROMPT_COLUMN].tolist()]
        batch_prompts = batch_df[PROMPT_COLUMN].tolist()
        if batch_df[OUTPUT_COLUMN].notna().any():
            continue  # Skip this batch
        
        try:
            results = asyncio.run(
                run_batch_inference_smmd(client, model, batch_prompts, **api_kwargs)
            )
            for i, r in enumerate(results):
                try:
                    df.at[batch_start_idx+i, OUTPUT_COLUMN] = pickle_response_object(r)
                except Exception as e:
                    print(f"Error in pickling at index {batch_start_idx+i}::{e}")
                    continue
        except Exception as e:
            print(f"Error at batch {batch_start_idx}::{e}")
        
        save_jsonl(df, output_path)
        time.sleep(3)
    save_jsonl(df, output_path)

def load_data(num_samples: int, seed: int = 42):
    path = "distillation_dataset.jsonl"
    dataset = load_dataset("json", data_files=path, split="train")
    dataset = dataset.shuffle(seed=seed)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    return dataset

def save_jsonl(data: pd.DataFrame, filename: str):
    data.to_json(filename, orient='records', lines=True)

def preprocess_example(
    example: DataSample,
) -> Message:
    """
    Preprocess a data sample into an OpenAI chat message format.
    Returns a list of dicts with 'role' and 'content' keys for system and user.
    """
    numbers: List[int] = example["nums"]
    target: int = example["target"]

    # ! Don't add in the system message for safety
    system_message, user_prompt = format_input(numbers, target)
    conversation = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]
    return conversation

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run OpenAI batch inference on Countdown dataset.")
    parser.add_argument('--model', type=str, default="o4-mini", help="Model name to use.")
    parser.add_argument('--output', type=str, default="o4-mini-distillation.jsonl", help="Output file path.")
    parser.add_argument('--num_samples', type=int, default=16000, help="Number of samples to process.")
    parser.add_argument('--batch_size', type=int, default=20, help="Batch size for inference.")
    parser.add_argument('--continue_from', type=str, default=None, help="Partially filled results file.")
    parser.add_argument('--reasoning_summary', type=bool, default=True, help="Set to True to get an auto summary and medium reasoning effort")
    return parser.parse_args()

if __name__ == "__main__":
    assert os.path.exists("distillation_dataset.jsonl"), ""

    _ = load_dotenv(".env", override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAIClient(api_key)

    args = get_args()
    print(f"Model: {args.model}")
    print(f"Output file: {args.output}")
    print(f"Num samples: {args.num_samples}")
    print(f"Batch size: {args.batch_size}")

    # Load and preprocess data
    if args.continue_from:
        df = pd.read_json(args.continue_from, lines=True)
        assert all([x in df.columns for x in (INPUT_COLUMN, OUTPUT_COLUMN, PROMPT_COLUMN)])
    else:
        dataset = load_data(args.num_samples)
        prompts = [preprocess_example(ex) for ex in dataset]
        input_samples = list(dataset)
        df = pd.DataFrame({
            INPUT_COLUMN: input_samples,
            PROMPT_COLUMN: prompts,
        })

    # Run engine
    api_kwargs = {}
    if args.reasoning_summary:
        print("Using auto reasoning summary.")
        api_kwargs["reasoning"] = {"effort": "medium", "summary": "auto"}
    engine(df, client, args.model, args.output, args.batch_size, **api_kwargs)