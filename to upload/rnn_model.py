import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        
        super().__init__()
        
        #TO-DO
        #1. Initialize Embedding Layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)

        #2. Initialize RNN layer
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers = n_layers,
            bidirectional = bidirectional,
            dropout = dropout
        )

        #3. Initialize a fully connected layer with Linear transformation
        self.fc = nn.Linear(2*hidden_dim,output_dim)

        #4. Initialize Dropout
        self.dropout = nn.Dropout(p = dropout)


        
    def forward(self, text, text_lengths):
        #text = [sent_len, batch_size]

        #TO-DO
        #1. Apply embedding layer that matches each word to its vector and apply dropout. Dim [sent_len, batch_size, emb_dim]
        embed = self.dropout(self.embedding(text))

        #2. Run the RNN along the sentences of length sent_len. #output = [sent len, batch size, hid dim * num directions]; #hidden = [num layers * num directions, batch size, hid dim]
        
        pkd_embed = nn.utils.rnn.pack_padded_sequence(embed,text_lengths)   # pack sequence

        pkd_out,hidden = self.rnn(pkd_embed)

        out, out_len = nn.utils.rnn.pad_packed_sequence(pkd_out)                # unpack sequence

        #3. Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers and apply dropout
        
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]),dim = 1))

        return self.fc(hidden)
