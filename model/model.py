import torch
import torch.nn as nn

from torch.nn.modules.module import Module
from .module.memory_bank import Memory_Unit_transformer
from .module.translayer import Transformer


class Inception_dxlstm(nn.Module):
    def __init__(self, input_size, out_size):
        super(Inception_dxlstm, self).__init__()

        # Multi-scale convolution layers
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                               stride=1, padding=1)
        self.conv2 = nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=5,
                               stride=1, padding=2)
        self.conv3 = nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=7,
                               stride=1, padding=3)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=out_size * 3, hidden_size=out_size, num_layers=1,
                            bidirectional=False, batch_first=True)

    def forward(self, x):
        # Multi-scale convolutions
        x1 = torch.relu(self.conv1(x.transpose(1, 2))).transpose(1, 2)
        x2 = torch.relu(self.conv2(x.transpose(1, 2))).transpose(1, 2)
        x3 = torch.relu(self.conv3(x.transpose(1, 2))).transpose(1, 2)

        # Concatenate multi-scale features along feature dimension
        x = torch.cat((x1, x2, x3), dim=-1)

        # LSTM layer
        x, _ = self.lstm(x)


        return x


class ADCLS_head(Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, out_dim), nn.Sigmoid())

    def forward(self, x):
        return self.mlp(x)


class FadNet(Module):
    def __init__(self, input_size, flag, a_nums, n_nums):
        super().__init__()
        self.flag = flag
        self.a_nums = a_nums
        self.n_nums = n_nums

        self.embedding = Inception_dxlstm(input_size, 1024)
        self.selfatt = Transformer(1024, 4, 8, 128, 512, dropout=0.5)
        self.cls_head = ADCLS_head(2048, 1)
        self.Amemory = Memory_Unit_transformer(nums=a_nums, dim=1024)
        self.Nmemory = Memory_Unit_transformer(nums=n_nums, dim=1024)

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        if self.flag == "Train":
            N_x = x[:b * n // 2]  #### Normal part
            A_x = x[b * n // 2:]  #### Abnormal part
            A_aug = self.Amemory(A_x)
            N_Aaug = self.Nmemory(A_x)
            A_Naug = self.Amemory(N_x)
            N_aug = self.Nmemory(N_x)

            x = torch.cat((x, (torch.cat([N_aug + A_Naug, A_aug + N_Aaug], dim=0))), dim=-1)
            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)

            return {
                "frame": pre_att,
            }
        else:
            A_aug = self.Amemory(x)
            N_aug = self.Nmemory(x)

            x = torch.cat([x, N_aug + A_aug], dim=-1)

            pre_att = self.cls_head(x).reshape((b, n, -1)).mean(1)
            return {"frame": pre_att}




if __name__ == "__main__":
    torch.cuda.set_device(1)  # set your gpu device
    m = FadNet(input_size=512, flag="Train", a_nums=60, n_nums=60).cuda()
    src = torch.rand(128, 200, 512).cuda()
    out = m(src)["frame"]

    print(out.size())
