import torch
import torch.nn as nn
from typing import List


class DRSA(nn.Module):
    """
    Deep Recurrent Survival Analysis model.
    A relatively shallow net, characterized by an LSTM layer followed by a Linear layer.
    """

    def __init__(
        self,
        n_features: int,
        hidden_dim: int,
        n_layers: int,
        embeddings: List[nn.Embedding],
        output_size: int = 1,
        LSTM_dropout: float = 0.0,
        Linear_dropout: float = 0.0,
    ):
        """
        inputs:
        * `n_features`
            - size of the input to the LSTM (number of features)
        * `hidden_dim`:
            - size (dimension) of the hidden state in LSTM
        * `n_layers`:
            - number of layers in LSTM
        * `embeddings`:
            - list of nn.Embeddings for each categorical variable
            - It is assumed the the 1st categorical feature corresponds with the 0th feature,
              the 2nd corresponds with the 1st feature, and so on.
        * `output_size`:
            - size of the linear layer's output, which should always be 1, unless altering this model
        * `LSTM_dropout`:
            - percent of neurons in LSTM layer to apply dropout regularization to during training
        * `Linear_dropout`:
            - percent of neurons in linear layer to apply dropout regularization to during training
        """
        super(DRSA, self).__init__()

        # hyper params
        self.n_features = n_features
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # embeddings
        self.embeddings = embeddings

        # model architecture
        self.lstm = nn.LSTM(
            sum([emb.embedding_dim for emb in self.embeddings])
            + self.n_features
            - len(self.embeddings),
            self.hidden_dim,
            self.n_layers,
            batch_first=True,
            dropout=LSTM_dropout,
        )
        self.fc = nn.Linear(hidden_dim, output_size)
        self.linear_dropout = nn.Dropout(p=Linear_dropout)
        self.sigmoid = nn.Sigmoid()
        
        # making sure embeddings get trained
        self.params_to_train = nn.ModuleList(self.embeddings)

    def forward(self, X: torch.tensor):
        """
        input:
        * `X`
            - input features of shape (batch_size, sequence length, self.n_features)
            
        output:
        * `out`: 
            - the DRSA model's predictions at each time step, for each observation in batch
            - out is of shape (batch_size, sequence_length, 1)
        """
        # concatenating embedding and numeric features
        all_embeddings = [
            emb(X[:, :, i].long()) for i, emb in enumerate(self.embeddings)
        ]

        print(all_embeddings[0].size())

        other_features = X[:, :, len(self.embeddings) :]
        all_features = torch.cat(all_embeddings + [other_features.float()], dim=-1)

        # passing input and hidden into model (hidden initialized as zeros)
        out, hidden = self.lstm(all_features.float())

        # passing to linear layer to reshape for predictions
        out = self.sigmoid(self.linear_dropout(self.fc(out)))

        return out
    
if __name__ == "__main__":
    # generating random data
    batch_size, seq_len, n_features = (64, 25, 10)
    data = torch.randn(batch_size, seq_len, n_features)

    # generating random embedding for each sequence
    n_embeddings = 10
    embedding_idx = torch.mul(
        torch.ones(batch_size, seq_len, 1),
        torch.randint(low=0, high=n_embeddings, size=(batch_size, 1, 1)),
    )

    # concatenating embeddings and features
    X = torch.cat([embedding_idx, data], dim=-1)

    print(X.size())

    # instantiating embedding parameters
    embedding_size = 5
    embeddings = torch.nn.Embedding(n_embeddings, embedding_size)

    # instantiating model
    model = DRSA(
        n_features=n_features + 1,  # +1 for the embeddings
        hidden_dim=2,
        n_layers=1,
        embeddings=[embeddings],
    )

    model(X)