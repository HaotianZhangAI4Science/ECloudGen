from torch import nn
import torch

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

        self.fc1 = nn.Linear(hidden_ch * 4 * 3 * 3 * 3, latent_dim*2)

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
        h5 = h5.view(-1,self.hidden_ch*4*3*3*3) #(B)(f)(3)(3)(3) -> (B)(f*3*3*3)
        return self.fc1(h5)
    
    def reparametrize(self, mu, logvar, factor=1.):
        std = logvar.mul(0.5).exp_().to(self.device)
        eps = torch.randn_like(std).to(self.device)
        return mu + (eps.mul(std) * factor) 

    def decode(self,z):
        re_con1 = self.leakyrelu(self.decoder_1(z)) # torch.Size([5, 13824])
        re_con1 = re_con1.view(-1, self.hidden_ch*4,3,3,3) #torch.Size([5, 512, 3, 3, 3])
        re_con2 = self.leakyrelu(self.bth_norm_6(self.decoder_2(re_con1))) #torch.Size([5, 512, 6, 6, 6])
        re_con3 = self.leakyrelu(self.bth_norm_7(self.decoder_3(re_con2))) #torch.Size([5, 256, 12, 12, 12])
        re_con4 = self.leakyrelu(self.bth_norm_8(self.decoder_4(re_con3))) #torch.Size([5, 128, 12, 12, 12])
        re_con5 = self.leakyrelu(self.bth_norm_9(self.decoder_5(re_con4))) #torch.Size([5, 32, 24, 24, 24])
        re_con6 = self.sigmoid(self.decoder_6(re_con5)) #torch.Size([5, 8, 24, 24, 24])
        return re_con6

    def forward(self, x, factor=1.):
        mu, logvar = self.encode(x).chunk(2,dim=-1)
        z = self.reparametrize(mu, logvar, factor=factor)
        recon = self.decode(z)
        return recon, mu, logvar

    def get_latent_var(self,x):
        mu, logvar = self.encode(x).chunk(2,dim=-1)
        z = self.reparametrize(mu, logvar)
        return z

