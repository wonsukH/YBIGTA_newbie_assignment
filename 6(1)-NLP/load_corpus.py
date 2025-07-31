from datasets import load_dataset # type: ignore

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # Rotten Tomatoes 데이터셋 사용 (더 많은 텍스트 데이터)
    dataset = load_dataset("rotten_tomatoes", split="train") 
    
    # 각 샘플의 텍스트를 코퍼스에 추가
    for sample in dataset:
        corpus.append(sample["text"])
    
    return corpus