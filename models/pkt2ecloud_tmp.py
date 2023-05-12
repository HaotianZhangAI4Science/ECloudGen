from torch import nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from torch.nn.utils.rnn import pack_padded_sequence

class Pkt2ECloud(nn.Module):
    def __init__(self, input_ch=8, hidden_ch=128, latent_dim=5, device='cpu'):
        super(Pkt2ECloud, self).__init__()
        self.input_ch = input_ch
        self.hidden_ch = hidden_ch
        self.latent_dim = latent_dim
        self.device = device

        # pocket encoder
        self.encoder_1 = nn.Conv3d(input_ch,32, 3, 1, 1)
        self.bth_norm_1 = nn.BatchNorm3d(32)
        self.encoder_2 = nn.Conv3d(32, 32, 3, 2, 1)
        self.bth_norm_2 = nn.BatchNorm3d(32)
        self.encoder_3 = nn.Conv3d(32, 64, 3, 1, 1)
        self.bth_norm_3 = nn.BatchNorm3d(64)
        self.encoder_4 = nn.Conv3d(64, hidden_ch * 4, 3, 2, 1)
        self.bth_norm_4 = nn.BatchNorm3d(hidden_ch * 4)
        self.encoder_5 = nn.Conv3d(hidden_ch * 4, hidden_ch * 4, 3, 2, 1)
        self.bth_norm_5 = nn.BatchNorm3d(hidden_ch * 4)

        self.fc1 = nn.Linear(hidden_ch * 4 * 3 * 3 * 3, latent_dim)
        self.fc2 = nn.Linear(hidden_ch * 4 * 3 * 3 * 3, latent_dim)

        # ECloud decoder
        self.decoder_1 =  nn.Linear(latent_dim, hidden_ch * 4 * 3 * 3 * 3)
        # CNN decoder
        self.decoder_2 = nn.ConvTranspose3d(hidden_ch * 4, hidden_ch * 4, 3, 2, padding=1, output_padding=1)
        self.bth_norm_6 = nn.BatchNorm3d(hidden_ch * 4, 1.e-3)
        self.decoder_3 = nn.ConvTranspose3d(hidden_ch * 4, hidden_ch * 2, 3, 2, padding=1, output_padding=1)
        self.bth_norm_7 = nn.BatchNorm3d(hidden_ch * 2, 1.e-3)
        self.decoder_4 = nn.Conv3d(hidden_ch * 2, hidden_ch, 3, 1, padding=1)
        self.bth_norm_8 = nn.BatchNorm3d(hidden_ch, 1.e-3)
        self.decoder_5 = nn.ConvTranspose3d(hidden_ch, 32, 3, 2, padding=1, output_padding=1)
        self.bth_norm_9 = nn.BatchNorm3d(32, 1.e-3)
        self.decoder_6 = nn.Conv3d(32, input_ch, 3, 1, padding=1)

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def encode(self, x):
        h1 = self.leakyrelu(self.bth_norm_1(self.encoder_1(x)))
        h2 = self.leakyrelu(self.bth_norm_2(self.encoder_2(h1)))
        h3 = self.leakyrelu(self.bth_norm_3(self.encoder_3(h2)))
        h4 = self.leakyrelu(self.bth_norm_4(self.encoder_4(h3)))
        h5 = self.leakyrelu(self.bth_norm_5(self.encoder_5(h4)))
        return self.fc1(h5)



class EncoderCNN(nn.Module):
    def __init__(self, in_layers):
        super(EncoderCNN, self).__init__()
        self.pool = nn.MaxPool3d((2, 2, 2))
        self.relu = nn.ReLU()
        layers = []
        out_layers = 32

        for i in range(8):
            layers.append(nn.Conv3d(in_layers, out_layers, 3, bias=False, padding=1))
            layers.append(nn.BatchNorm3d(out_layers))
            layers.append(self.relu)
            in_layers = out_layers
            if (i + 1) % 2 == 0:
                # Duplicate number of layers every alternating layer.
                out_layers *= 2
                layers.append(self.pool)
        layers.pop()  # Remove the last max pooling layer!
        self.fc1 = nn.Linear(256, 512)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=2).mean(dim=2).mean(dim=2)
        x = self.relu(self.fc1(x))
        return x
        
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, captions, lengths):
        """Decode shapes feature vectors and generates SMILES."""
        embeddings = self.embed(captions) ##torch.Size([5, 52, 512])
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1) #torch.Size([5, 53, 512])
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) #torch.Size([260, 512]) #torch.Size([52])
        hiddens, _ = self.lstm(packed) #torch.Size([260, 512]) #torch.Size([52])
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples SMILES tockens for given shape features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids

    def sample_prob(self, features, states=None):
        """Samples SMILES tockens for given shape features (probalistic picking)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(62):  # maximum sampling length
            hiddens, states = self.lstm(inputs, states)
            outputs = self.linear(hiddens.squeeze(1))
            if i == 0:
                predicted = outputs.max(1)[1]
            else:
                probs = F.softmax(outputs, dim=1)

                # Probabilistic sample tokens
                if probs.is_cuda:
                    probs_np = probs.data.cpu().numpy()
                else:
                    probs_np = probs.data.numpy()

                rand_num = np.random.rand(probs_np.shape[0])
                iter_sum = np.zeros((probs_np.shape[0],))
                tokens = np.zeros(probs_np.shape[0], dtype=np.int)

                for i in range(probs_np.shape[1]):
                    c_element = probs_np[:, i]
                    iter_sum += c_element
                    valid_token = rand_num < iter_sum
                    update_indecies = np.logical_and(valid_token,
                                                     np.logical_not(tokens.astype(np.bool)))
                    tokens[update_indecies] = i

                # put back on the GPU.
                if probs.is_cuda:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)).cuda())
                else:
                    predicted = Variable(torch.LongTensor(tokens.astype(np.int)))

            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)
        return sampled_ids