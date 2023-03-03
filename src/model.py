from torch import nn
import torch

class Encoder(nn.Module):
    def __init__(self, in_channels, koopman_size):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        self.conv5 = nn.Conv1d(in_channels=256, out_channels=koopman_size, kernel_size=5, padding=2)
        self.bn5 = nn.BatchNorm1d(koopman_size)
        self.lstm = nn.LSTM(input_size=koopman_size, hidden_size=koopman_size, num_layers=1, batch_first=True)
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
      out = self.leaky_relu(self.bn1(self.conv1(x)))
      out = self.leaky_relu(self.bn2(self.conv2(out)))
      out = self.leaky_relu(self.bn3(self.conv3(out)))
      out = self.leaky_relu(self.bn4(self.conv4(out)))
      out = self.leaky_relu(self.bn5(self.conv5(out)))
      out = out.permute(0, 2, 1)
      out, _ = self.lstm(out)
      out = out.permute(0, 2, 1)
      return out

class Decoder(nn.Module):
  def __init__(self, koopman_size, out_channels, output_length):
    super(Decoder, self).__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=koopman_size, out_channels=256, kernel_size=4, padding=3, dilation=2)
    self.bn1 = nn.BatchNorm1d(256)
    self.conv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, padding=3, dilation=2)
    self.bn2 = nn.BatchNorm1d(128)
    self.conv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, padding=3, dilation=2)
    self.bn3 = nn.BatchNorm1d(64)
    self.conv4 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, padding=3, dilation=2)
    self.bn4 = nn.BatchNorm1d(32)
    self.conv5 = nn.ConvTranspose1d(in_channels=32, out_channels=out_channels, kernel_size=4, padding=3, dilation=2)
    self.linear = nn.LazyLinear(output_length)
    self.leaky_relu = nn.LeakyReLU()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    out = self.leaky_relu(self.bn1(self.conv1(x)))
    out = self.leaky_relu(self.bn2(self.conv2(out)))
    out = self.leaky_relu(self.bn3(self.conv3(out)))
    out = self.leaky_relu(self.bn4(self.conv4(out)))
    out = self.leaky_relu((self.conv5(out)))
    out = self.sigmoid(self.linear(out))
    return out

class Regressor(nn.Module):
  def __init__(self, koopman_size):
    super(Regressor, self).__init__()
    self.conv1 = nn.ConvTranspose1d(in_channels=koopman_size, out_channels=256, kernel_size=4, stride=1, padding=0)
    self.bn1 = nn.BatchNorm1d(256)
    self.conv2 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=4, stride=1, padding=0)
    self.bn2 = nn.BatchNorm1d(128)
    self.conv3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=4, stride=1, padding=0)
    self.bn3 = nn.BatchNorm1d(64)
    self.conv4 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=4, stride=1, padding=0)
    self.bn4 = nn.BatchNorm1d(32)
    self.conv5 = nn.ConvTranspose1d(in_channels=32, out_channels=1, kernel_size=4, stride=1, padding=0)
    self.linear = nn.LazyLinear(1)
    self.leaky_relu = nn.LeakyReLU()

  def forward(self, x):
    out = self.leaky_relu(self.bn1(self.conv1(x)))
    out = self.leaky_relu(self.bn2(self.conv2(out)))
    out = self.leaky_relu(self.bn3(self.conv3(out)))
    out = self.leaky_relu(self.bn4(self.conv4(out)))
    out = self.leaky_relu((self.conv5(out)))
    out = self.linear(out.reshape(out.shape[0], -1))
    return out

class PInverse(nn.Module):
  def __init__(self):
    super(PInverse, self).__init__()

  def forward(self, z_p):
    return torch.pinverse(z_p)

class EncoderDecoder(nn.Module):
  def __init__(self, in_channels, koopman_size, output_length):
    super(EncoderDecoder, self).__init__()
    self.encoder = Encoder(in_channels, koopman_size)
    self.decoder = Decoder(koopman_size, in_channels, output_length)
    self.pinverse = PInverse()
    self.regressor = Regressor(koopman_size)

  def forward(self, x):
    z = self.encoder(x)

    z_p = z[:,:,:-1]
    z_f = z[:,:,1:]

    z_p_inverse = self.pinverse(z_p.permute(0, 2, 1).reshape(-1, z.shape[1]))

    C = torch.matmul(z_p_inverse, z_f.permute(0, 2, 1).reshape(-1, z.shape[1]))

    z_hat_f = torch.matmul(z_p.permute(0, 2, 1), C).permute(0, 2, 1)

    x_f = x[:,:,1:]
    x_hat_f = self.decoder(nn.functional.pad(z_hat_f, (1, 0)))[:,:,1:]

    reg_hat = self.regressor(nn.functional.pad(z_hat_f, (1, 0)))
    regression = self.regressor(z)

    rec = self.decoder(z)
    
    return (rec, x), (z_f, z_hat_f), (x_f, x_hat_f), (regression, reg_hat)
