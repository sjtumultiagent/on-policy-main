from torch.utils.tensorboard import SummaryWriter

from onpolicy.algorithms.utils.rnn import RNNLayer

writer = SummaryWriter("E:\code\drone\on-policy-main\src\log")
writer.add_graph(rnn, )

rnn = RNNLayer(64, 64, 1, True)

print(rnn)
