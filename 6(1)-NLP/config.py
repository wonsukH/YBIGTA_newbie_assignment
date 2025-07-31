from typing import Literal


device = "cpu"
d_model = 512  

# Word2Vec - 더 강력한 학습을 위해 조정
window_size = 3  
method: Literal["cbow", "skipgram"] = "skipgram"  
lr_word2vec = 1e-03  
num_epochs_word2vec = 12 

# GRU - 더 강력한 분류 성능을 위해 조정
hidden_size = 512  
num_classes = 4
lr = 2e-03 
num_epochs = 200 
batch_size = 4 