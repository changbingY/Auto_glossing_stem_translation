import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence#, masked_cross_entropy


USE_CUDA = torch.cuda.is_available()
PAD_TOKEN = 0
UNK_TOKEN = 1
START_TOKEN = 2
END_TOKEN = 3


class Attn(nn.Module):
    def __init__(self,method, hidden_size):
        super(Attn, self).__init__()
        
        #self.method = method
        self.hidden_size = hidden_size
        
        #if self.method == 'general':
        #    self.attn = nn.Linear(self.hidden_size, hidden_size)

        #if self.method == 'concat':
        #self.attn = nn.Linear(self.hidden_size +768, hidden_size)
        self.attn_source = nn.Linear(self.hidden_size * 2, hidden_size)
        self.attn_trans = nn.Linear(self.hidden_size +768, hidden_size)
        #self.v = nn.Parameter(torch.FloatTensor(hidden_size))
        #self.v = nn.Linear(self.hidden_size, 1)
        self.v_source = nn.Linear(self.hidden_size, 1)
        #self.v = nn.Linear(hidden_size, 1)
        self.v_trans = nn.Linear(self.hidden_size, 1)

    def forward(self, hidden, encoder_outputs,attn_type="source"):
        """Attention computation.
        Args:
            hidden: tensor [1, batch_size, hidden_size]
            encoder_outputs: [batch_size, sequence_length, hidden_size]
        Return:
            attn_energies: [batch_size, 1, hidden_size]
        """
        max_len = encoder_outputs.size(1)
        this_batch_size = encoder_outputs.size(0)

        # Create variable to store attention energies
        #attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        #if USE_CUDA:
         #   attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        #for b in range(this_batch_size):
            # Calculate energy for each encoder output
           # for i in range(max_len):
             #   attn_energies[b, i] = self.score(hidden[:, b].squeeze(0), encoder_outputs[b, i])
        hidden = hidden.squeeze(0).unsqueeze(1).repeat((1, max_len, 1))
        if attn_type == "source":
            attn_energies = self.attn_source(torch.cat((hidden, encoder_outputs), dim=2))
            attn_energies = self.v_source(attn_energies).squeeze(2)
        elif attn_type == "trans":
            attn_energies = self.attn_trans(torch.cat((hidden, encoder_outputs), dim=2))
            attn_energies = self.v_trans(attn_energies).squeeze(2)
        #attn_energies = self.attn(torch.cat((hidden, encoder_outputs), dim=2))
        #attn_energies = self.v(attn_energies).squeeze(2)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies
        #return F.softmax(attn_energies,dim=1).unsqueeze(1)
    
    #def score(self, hidden, encoder_output):
        
      #  if self.method == 'dot':
      #      energy = hidden.dot(encoder_output)
      #      return energy
        
      #  elif self.method == 'general':
       #     energy = self.attn(encoder_output)
       #     energy = hidden.dot(energy)
       #     return energy
        
       # elif self.method == 'concat':
       #     energy = self.attn(torch.cat((hidden, encoder_output)))
       #     energy = self.v.dot(energy)
       #     return energy


class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=PAD_TOKEN)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        #self.combine = nn.Linear(hidden_size * 2, hidden_size)
        self.combine_input = nn.Linear(hidden_size * 2, hidden_size)
        #self.combine_input = nn.Linear(512 * 2, 512)
        self.combine_context = nn.Linear(hidden_size +768, hidden_size)
        #self.combine_context = nn.Linear(512 * 2,512)
        self.out = nn.Linear(hidden_size, output_size)
        
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)
    
    def forward(self, input_seq, input_aux, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = self.combine_input(torch.cat((embedded, input_aux), dim=1))
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        if last_hidden is None:
            last_hidden = (torch.zeros_like(embedded),torch.zeros_like(embedded))
            #last_hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size).cuda()

        rnn_output, (hidden,cell) = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        #attn_weights = self.attn(rnn_output, encoder_outputs)
        #context = attn_weights.bmm(encoder_outputs) # B x S=1 x N
        assert isinstance(encoder_outputs, list) and len(encoder_outputs) == 2
        contexts = []
        attention_energies = []
        for i, enc_outputs in enumerate(encoder_outputs):
            if i == 0:
                attn_type = "source"
            elif i == 1:
                attn_type = "trans"
            #attn_weights = self.attn(rnn_output, enc_outputs, attn_type)
            attn_energies = self.attn(rnn_output, enc_outputs, attn_type)
            attn_weights = F.softmax(attn_energies,dim=1).unsqueeze(1)
            #attn_weights = self.attn(rnn_output, enc_outputs)
            context = attn_weights.bmm(enc_outputs) # B x S=1 x N
            contexts.append(context.squeeze(1))
            attention_energies.append(attn_energies)
        contexts = torch.cat(contexts, dim=1)
        context = self.combine_context(contexts)
        
        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        #context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, (hidden,cell), attention_energies
