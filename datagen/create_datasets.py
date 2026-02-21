from datasets import load_dataset

def load_and_split_countdown3to4(seed=42, split_size=16000):
    ds = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train")
    ds = ds.shuffle(seed=seed)
    first = ds.select(range(0, split_size))
    second = ds.select(range(split_size, 2*split_size))
    third = ds.select(range(2*split_size, 3*split_size))
    return first, second, third

if __name__ == "__main__":
    first, second, third = load_and_split_countdown3to4()

    first.to_pandas().to_json("distillation_dataset.jsonl", lines=True, orient='records')
    second.to_pandas().to_json("eval_dataset.jsonl", lines=True, orient='records')
    third.to_pandas().to_json("rlvr_dataset.jsonl", lines=True, orient='records')