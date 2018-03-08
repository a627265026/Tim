import torch
from torch import nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

#  以横轴坐标选取时间点

TIME_STEP = 10
INPUT_SIZE = 1
LR = 0.02


steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)
y_np = np.cos(steps)
plt.plot(steps, y_np, 'r-', label='target(cos)')
plt.plot(steps, x_np, 'b-', label='input(sin)')
plt.legend(loc='best')
plt.show()


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=32,     # 每个隐藏层中神经元的个数
            num_layers=1,       # 隐藏层的个数
            batch_first=True    # 输入/输出 将把 batch 放在第一位
                                # e.g(batch, time_step, input_size)
        )
        self.out = nn.Linear(32, 1)

    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # r_out (batch, time_step, hidden_size)
        r_out, h_state = self.rnn(x, h_state)
        r_out = r_out.view(-1, 32)
        # view 的作用
        # 保持数据不变的情况下重塑变量格式
        # x = torch.randn(2, 4)
        # >>> x = 1.5600 -1.6180 -2.0366 2.7115
        #         0.8415 -1.0103 -0.4793 1.5734
        #         [torch.FloatTensor of size 2x4]
        # x = x.view(4, 2)
        # >>> x =  1.5600 -1.6180
        # -2.0366  2.7115
        # 0.8415 -1.0103
        # -0.4793  1.5734
        # [torch.FloatTensor of size 4x2]
        # 当 某个参数为 -1 时, 该参数取决于 Tensor的大小和其他参数 自动赋值
        outs = self.out(r_out)
        return outs, h_state


rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.MSELoss()
plt.figure(1, figsize=(12, 5))
plt.ion()

h_state = None
for step in range(60):
    start, end = step * np.pi, (step+1) * np.pi

    steps = np.linspace(start, end, TIME_STEP, dtype=np.float32)
    x_np = np.sin(steps)
    y_np = np.cos(steps)
    x = Variable(torch.from_numpy(x_np[np.newaxis, :, np.newaxis]))
    # shape ( batch, tmie_step, input_size)
    # np.newaxis 用来创建新的列
    y = Variable(torch.from_numpy(y_np[np.newaxis, :, np.newaxis]))

    prediction, h_state = rnn(x, h_state)
    h_state = Variable(h_state.data)
    # repack the hidden state, break the  connection  from last iteration
    loss = loss_func(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()
