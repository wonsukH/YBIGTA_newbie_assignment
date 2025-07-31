import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam

from transformers import PreTrainedTokenizer

from typing import Literal

# 구현하세요!


class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method
        # 구현하세요!
        self.vocab_size = vocab_size
        self.d_model = d_model

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: list[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)
        # 구현하세요!
        print(f"Starting Word2Vec training with {len(corpus)} texts...")
        
        for epoch in range(num_epochs):
            total_loss = 0
            num_batches = 0
            
            print(f"Epoch {epoch+1}/{num_epochs} - Processing texts...")
            
            for i, text in enumerate(corpus):
                if i % 50 == 0:  # 50개마다 진행상황 표시
                    print(f"  Processing text {i+1}/{len(corpus)}")
                
                # 텍스트를 토크나이저
                tokens = tokenizer.encode(text, add_special_tokens=False)
                
                if len(tokens) < 2 * self.window_size + 1:
                    continue
                
                if self.method == "cbow":
                    self._train_cbow(tokens, tokenizer, criterion, optimizer)
                elif self.method == "skipgram":
                    self._train_skipgram(tokens, tokenizer, criterion, optimizer)
                
                num_batches += 1
            
            print(f"Epoch {epoch+1}/{num_epochs} completed.")
        
        print("Word2Vec training completed!")

    def _train_cbow(
        self,
        # 구현하세요!
        tokens: list[int],
        tokenizer: PreTrainedTokenizer,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        # CBOW: 주변 단어들로 중심 단어 예측 (배치 처리)
        batch_inputs = []
        batch_targets = []
        
        for i in range(self.window_size, len(tokens) - self.window_size):
            # 중심 단어
            target = tokens[i]
            
            # 주변 단어들 (컨텍스트)
            context = []
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i:
                    context.append(tokens[j])
            
            if len(context) == 0:
                continue
            
            # 컨텍스트 단어들의 임베딩 평균
            context_tensor = torch.tensor(context, dtype=torch.long)
            context_embeddings = self.embeddings(context_tensor)
            context_mean = context_embeddings.mean(dim=0, keepdim=True)
            
            batch_inputs.append(context_mean.squeeze(0))
            batch_targets.append(target)
        
        if not batch_inputs:
            return
        
        # 배치로 한 번에 처리
        batch_inputs = torch.stack(batch_inputs)
        batch_targets = torch.tensor(batch_targets, dtype=torch.long)
        
        # 예측
        logits = self.weight(batch_inputs)
        
        # 손실 계산 및 역전파
        loss = criterion(logits, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def _train_skipgram(
        self,
        # 구현하세요!
        tokens: list[int],
        tokenizer: PreTrainedTokenizer,
        criterion: nn.CrossEntropyLoss,
        optimizer: Adam
    ) -> None:
        # 구현하세요!
        # Skip-gram: 중심 단어로 주변 단어들 예측 (배치 처리)
        batch_inputs = []
        batch_targets = []
        
        for i in range(self.window_size, len(tokens) - self.window_size):
            # 중심 단어
            center_word = tokens[i]
            
            # 주변 단어들
            for j in range(i - self.window_size, i + self.window_size + 1):
                if j != i and 0 <= j < len(tokens):
                    target = tokens[j]
                    
                    batch_inputs.append(center_word)
                    batch_targets.append(target)
        
        if not batch_inputs:
            return
        
        # 배치로 한 번에 처리
        batch_inputs = torch.tensor(batch_inputs, dtype=torch.long)
        batch_targets = torch.tensor(batch_targets, dtype=torch.long)
        
        # 중심 단어의 임베딩
        center_embeddings = self.embeddings(batch_inputs)
        
        # 예측
        logits = self.weight(center_embeddings)
        
        # 손실 계산 및 역전파
        loss = criterion(logits, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # 구현하세요!
