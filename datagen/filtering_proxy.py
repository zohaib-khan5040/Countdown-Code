import re
import json
from typing import Dict, Any, List

import pandas as pd
from datasets import load_dataset, Dataset

import rewards

def load_json_from_response(text: str):
    # Try to find JSON after </think>
    match = re.search(r"</think>\s*({.*})\s*$", text, re.DOTALL)
    if match:
        answer_content = match.group(1).strip()
        out_json = json.loads(answer_content)
        return out_json
    # Fallback: find any JSON object in the text
    match_any = re.search(r"({[^{}]*\{[^{}]*\}[^{}]*}|{.*})", text, re.DOTALL)
    if match_any:
        answer_content = match_any.group(1).strip()
        out_json = json.loads(answer_content)
        return out_json
    return None

def format_model_output(model_output: Dict[str, Any]) -> str:
    if pd.isna(model_output):
        return None
    model_text: str = model_output['text']  # contains the <think> tags
    
    # Extract the summary
    list_of_summary_objs: List[str] = model_output['summary']
    stitched_summary = '\n'.join(list_of_summary_objs)
    # Extract all content inside <think> tags
    think_match = re.search(r'<think>(.*?)</think>', model_text, re.DOTALL)
    final_cot = think_match.group(1).strip() if think_match else ''
    # Extract the json object
    json_obj = json.dumps(load_json_from_response(model_text))

    think_content = f'{stitched_summary}\n\n{final_cot}'
    think_content = think_content.replace("<think>", "").replace("</think", "")
    distillation_trace = f"<think>{think_content}</think>\n{json_obj}"
    return distillation_trace


if __name__ == "__main__":
    ds = load_dataset("json", data_files="o4-mini-distillation-16k.jsonl", split="train")
    df = ds.to_pandas()

    df['response'] = df['output'].apply(format_model_output)
    df['query'] = df['prompt'].apply(lambda listt: listt[0]['content'])

    df['equation_reward'] = df.apply(
        lambda row: rewards.equation_reward_func(nums=row['input']['nums'], 
                                                 target=row['input']['target'], 
                                                 response_obj=row['response']),
        axis=1
    )
    df['test_reward'] = df['response'].apply(rewards.test_pass_reward_func)

    # Filter for proxy reward being satisfied and correctly formatted responses
    filtered_df = df[(df['test_reward'] > 0)][["input", "query", "response"]]

    ds = Dataset.from_pandas(filtered_df)
    ds.to_json("sft_dataset.jsonl")

if __name__ == "__main__":
    # Load raw distillation data
    ds = load_dataset("json", data_files="o4-mini-distillation-16k.jsonl", split="train")
    df = ds.to_pandas()

    # Formatting logic
    df['response_text'] = df['output'].apply(format_model_output)
    df['query_text'] = df['prompt'].apply(lambda x: x[0]['content'] if isinstance(x, list) else x)

    # 1. Calculate Rewards for Monitoring
    # We compute both to distinguish between 'Proxy Success' and 'True Success'
    df['test_reward'] = df['response_text'].apply(rewards.test_pass_reward_func)
    df['equation_reward'] = df.apply(
        lambda row: rewards.equation_reward_func(
            nums=row['input']['nums'], 
            target=row['input']['target'], 
            response_obj=row['response_text']
        ), axis=1
    )

    # 2. Outcome-Based Filtering (SFT Catalyst)
    # We keep all trajectories where the test passed (Proxy Reward = 1)
    # This intentionally includes 'cheating' samples (~1.2% in your data)
    filtered_df = df[df['test_reward'] > 0].copy()

    # 3. Structure for verl
    # Adding 'equation_reward' to extra_info allows you to log it as a custom metric
    filtered_df['extra_info'] = filtered_df.apply(
        lambda row: {
            "question": row['query_text'],
            "answer": row['response_text'],
            "ground_truth": {
                "numbers": row['input']['nums'],
                "target": row['input']['target'],
                "equation_reward": row['equation_reward']
            }
        }, axis=1
    )

    # Save to Parquet
    final_df = filtered_df[['extra_info']]
    train_df = final_df.sample(frac=0.9, random_state=42)
    test_df = final_df.drop(train_df.index)

    train_df.to_parquet("data/sft/train.parquet", index=False)
    test_df.to_parquet("data/sft/test.parquet", index=False)
    
    print(f"Dataset saved. Mean Equation Reward in SFT data: {filtered_df['equation_reward'].mean():.4f}")