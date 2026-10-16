
import torch
import torch.nn as nn


class NextWordModel(nn.Module):
    """Autoregressive next-word prediction using an RNN-family model."""

    def __init__(
        self,
        vocab_size,
        architecture="lstm",
        embedding_dim=64,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
    ):
        super().__init__()

        architecture = architecture.lower()

        if architecture not in {"rnn", "birnn", "lstm", "gru"}:
            raise ValueError(f"Unsupported architecture: {architecture}")

        self.architecture = architecture
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Recurrent layers
        if architecture in {"rnn", "birnn"}:
            self.recurrent = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=dropout if num_layers > 1 else 0.0,
                bidirectional=(architecture == "birnn"),
            )
        elif architecture == "lstm":
            self.recurrent = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
        else:
            self.recurrent = nn.GRU(
                input_size=embedding_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )

        directions = 2 if architecture == "birnn" else 1

        self.output_layer = nn.Linear(
            hidden_dim * directions,
            vocab_size,
        )

    def forward(self, input_ids):
        """
        input_ids: [batch_size, sequence_length]
        returns:   [batch_size, vocab_size] logits for the next word
        """
        embedded = self.embedding(input_ids)

        if self.architecture == "lstm":
            _, (hidden, _) = self.recurrent(embedded)
        else:
            _, hidden = self.recurrent(embedded)

        if self.architecture == "birnn":
            # Final forward and backward states.
            # Both use only the supplied input sequence.
            final_forward = hidden[-2]
            final_backward = hidden[-1]
            final_state = torch.cat(
                [final_forward, final_backward], dim=1
            )
        else:
            final_state = hidden[-1]

        logits = self.output_layer(final_state)
        return logits


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


if __name__ == "__main__":
    vocab_size = 13486
    batch_size = 4
    sequence_length = 24

    sample_input = torch.randint(
        0, vocab_size, (batch_size, sequence_length)
    )

    for architecture in ["rnn", "birnn", "lstm", "gru"]:
        model = NextWordModel(
            vocab_size=vocab_size,
            architecture=architecture,
        )

        logits = model(sample_input)

        print(
            f"{architecture.upper():5s} | "
            f"Output shape: {tuple(logits.shape)} | "
            f"Parameters: {count_parameters(model):,}"
        )

        assert logits.shape == (batch_size, vocab_size)
