import os
from datasets import load_dataset
from prompts import format_input as prompt_fn

# --- CONFIGURATION ---
# Change this to point to your local .jsonl file
jsonl_file_path = "rlvr_dataset.jsonl" 
# Example structure in jsonl: {"nums": [1, 2, 3], "target": 6}

def make_map_fn(split):
    def process_fn(example, idx):
        # We assume the keys 'nums' and 'target' exist in your JSONL
        numbers = example['nums']
        target = example['target']
        
        system_message, user_prompt = prompt_fn(numbers, target)

        data = {
            "data_source": "local_jsonl",
            "prompt": [
                {"role": "system" ,"content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "target": target, 
                    "numbers": numbers
                }
            },
            "extra_info": {
                'split': split,
                'index': idx
            }
        }
        return data

    return process_fn


if __name__ == "__main__":
    # 1. Load the dataset from local JSONL
    print(f"Loading dataset from {jsonl_file_path}...")
    
    # 'json' builder handles .jsonl files automatically
    # split='train' loads all data into a single split named 'train'
    ds = load_dataset("json", data_files=jsonl_file_path, split='train')

    # 2. Sample (Optional)
    SAMPLE_SIZE = 5000
    if len(ds) > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} examples from {len(ds)} total...")
        ds = ds.shuffle(seed=42).select(range(SAMPLE_SIZE))
    else:
        print(f"Dataset has {len(ds)} examples (fewer than {SAMPLE_SIZE}), using all.")

    # 3. Perform train-test split
    print("Splitting dataset...")
    split_ds = ds.train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds['train']
    test_ds = split_ds['test']

    print(f"Train size: {len(train_ds)}")
    print(f"Test size: {len(test_ds)}")

    # 4. Apply the mapping function
    original_cols = ds.column_names
    
    print("Mapping training set...")
    train_dataset = train_ds.map(
        make_map_fn('train'), 
        with_indices=True,
        remove_columns=original_cols
    )
    
    print("Mapping test set...")
    test_dataset = test_ds.map(
        make_map_fn('test'), 
        with_indices=True,
        remove_columns=original_cols
    )

    # 5. Save to parquet
    out_dir = "data/rlvr"
    os.makedirs(out_dir, exist_ok=True)

    train_path = os.path.join(out_dir, 'train.parquet')
    test_path = os.path.join(out_dir, 'test.parquet')

    print(f"Saving to {out_dir}...")
    train_dataset.to_parquet(train_path)
    print(f"Exported train dataset of size {len(train_dataset)}")
    test_dataset.to_parquet(test_path)
    print(f"Exported test dataset of size {len(test_dataset)}")
    
    print("Done!")