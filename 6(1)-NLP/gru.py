import torch
from torch import nn, Tensor


class GRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        # 구현하세요!
        self.hidden_size = hidden_size
        
        # GRU 게이트들을 위한 선형 레이어들 (효율적인 구현)
        self.r_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.z_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.h_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        # 구현하세요!
        # x: (batch_size, input_size), h: (batch_size, hidden_size)
        batch_size = x.size(0)
        
        # 입력과 이전 hidden state 결합 (효율적인 연산)
        combined = torch.cat([x, h], dim=1)  # (batch_size, input_size + hidden_size)
        
        # Reset gate
        r = torch.sigmoid(self.r_gate(combined))  # (batch_size, hidden_size)
        
        # Update gate
        z = torch.sigmoid(self.z_gate(combined))  # (batch_size, hidden_size)
        
        # Candidate hidden state (효율적인 계산)
        h_candidate = torch.tanh(self.h_gate(torch.cat([x, r * h], dim=1)))
        
        # New hidden state
        h_new = (1 - z) * h + z * h_candidate
        
        return h_new


class GRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUCell(input_size, hidden_size)
        # 구현하세요!

    def forward(self, inputs: Tensor) -> Tensor:
        # 구현하세요!
        # inputs: (batch_size, sequence_length, input_size)
        batch_size, seq_len, input_size = inputs.size()
        
        # 초기 hidden state (효율적인 초기화)
        h = torch.zeros(batch_size, self.hidden_size, device=inputs.device, dtype=inputs.dtype)
        
        # 시퀀스를 순차적으로 처리 (효율적인 루프)
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # (batch_size, input_size) - 현재 단어
            h = self.cell(x_t, h)  # GRUCell 호출
        
        # 마지막 hidden state 반환
        return h