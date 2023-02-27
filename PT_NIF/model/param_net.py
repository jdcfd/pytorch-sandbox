"""
# Build "parameter net"
            para_net = t # tf.concat([t],axis=-1)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, N_t, activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, 1,  activation=ACT,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
            para_net = tf.layers.dense(para_net, Total_para, activation=None,
                                       kernel_initializer=init_weight, bias_initializer=init_weight)
"""
from torch import nn
import torch.nn.functional as F

class ParamNet(nn.Module):
    def __init__(self,act_fn=nn.SiLU,L_s=3,N_in=1,N_t=2,N_s=2):
        super(ParamNet, self).__init__()
        self.tot_params = (L_s-1)*(N_s+1)*N_s + 3*N_s + 1
        self.linear_stack = nn.Sequential(
            nn.Linear(N_in, N_t),
            act_fn(),
            nn.Linear(N_t, N_t),
            act_fn(),
            nn.Linear(N_t, 1),
            act_fn(),
            nn.Linear(1, self.tot_params),
        )

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits