# Parts of this code are based on https://github.com/mveres01/pytorch-drl4vrp/blob/master/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.embed = nn.Linear(input_size, hidden_size)
        self.embed_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, input):
        output = F.relu(self.embed(input))
        output = self.embed_2(output)
        return output


class Attention(nn.Module):
    def __init__(self, device, hidden_size):
        super(Attention, self).__init__()

        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.W = nn.Parameter(torch.zeros((1, hidden_size, hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, static_hidden):
        batch_size, hidden_size, _ = static_hidden.size()

        v = self.v.expand(batch_size, 1, hidden_size)
        W = self.W.expand(batch_size, hidden_size, -1)

        attns = torch.bmm(v, torch.tanh(torch.bmm(W, static_hidden)))
        attns = F.softmax(attns, dim=2)
        return attns


class Pointer(nn.Module):

    def __init__(self, device, hidden_size):
        super(Pointer, self).__init__()
        
        self.v = nn.Parameter(torch.zeros((1, 1, hidden_size),
                                          device=device, requires_grad=True))

        self.encoder_attn = Attention(device, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, all_hidden):
        enc_attn = self.encoder_attn(all_hidden)
        context = enc_attn.bmm(all_hidden.permute(0, 2, 1))

        input = context.squeeze(1)

        output = F.relu(self.fc1(input))
        output = F.relu(self.fc2(output))

        output = output.unsqueeze(2)
        output = output.expand_as(all_hidden)

        v = self.v.expand(all_hidden.size(0), -1, -1)
        probs = torch.bmm(v, torch.tanh(all_hidden + output)).squeeze(1)

        return probs


class HPCActorModel(nn.Module):
    def __init__(self, device, hidden_size=128):
        super(HPCActorModel, self).__init__()

        self.all_embed = Encoder(5, hidden_size)
        self.pointer = Pointer(device, hidden_size)
        self.origin_embed = Encoder(5, hidden_size)

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static_input, dynamic_input_float):
        # Set the input feature values of already visited customers (demand == 0) to zero
        active_inputs = dynamic_input_float[:, 1:, 1] > 0
        static_input[:, 1:, :] = static_input[:, 1:, :] * active_inputs.unsqueeze(2).float()

        # Embed inputs
        all_hidden = self.all_embed.forward(
            torch.cat((static_input, dynamic_input_float), dim=2))

        probs = self.pointer.forward(all_hidden.permute(0, 2, 1))
        return probs
