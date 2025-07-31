from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    dataset = load_dataset("ag_news", split="train") 
    
    # 각 샘플의 텍스트를 코퍼스에 추가
    for sample in dataset:
        corpus.append(sample["text"])
    
    return corpus