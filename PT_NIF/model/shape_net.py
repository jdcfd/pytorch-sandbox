import torch
from torch import nn
import torch.nn.functional as F

"""
# Distribute to weight and biases
            weight_1 = para_net[:,0:N_s];                           weight_1 = tf.reshape(weight_1, shape=[-1, 1,  N_s])
            weight_2 = para_net[:,N_s:((N_s+1)*N_s)];               weight_2 = tf.reshape(weight_2, shape=[-1, N_s, N_s])
            weight_3 = para_net[:,(N_s**2+N_s):(2*N_s**2+N_s)];     weight_3 = tf.reshape(weight_3, shape=[-1, N_s, N_s])
            weight_4 = para_net[:,(2*N_s**2+N_s):(2*N_s**2+2*N_s)]; weight_4 = tf.reshape(weight_4, shape=[-1, N_s, 1 ])
            bias_1   = para_net[:,(2*N_s**2+2*N_s):(2*N_s**2+3*N_s)]; bias_1   = tf.reshape(bias_1,   shape=[-1, N_s])
            bias_2   = para_net[:,(2*N_s**2+3*N_s):(2*N_s**2+4*N_s)]; bias_2   = tf.reshape(bias_2,   shape=[-1, N_s])
            bias_3   = para_net[:,(2*N_s**2+4*N_s):(2*N_s**2+5*N_s)]; bias_3   = tf.reshape(bias_3,   shape=[-1, N_s])
            bias_4   = para_net[:,(2*N_s**2+5*N_s):];                 bias_4   = tf.reshape(bias_4,   shape=[-1, 1])

            # Build "shape net"
            u = ACT(tf.einsum('ai,aij->aj', x, weight_1) + bias_1)
            u = ACT(tf.einsum('ai,aij->aj', u, weight_2) + bias_2) + u
            u = ACT(tf.einsum('ai,aij->aj', u, weight_3) + bias_3) + u
            u = tf.einsum('ai,aij->aj',u, weight_4) + bias_4
"""

def ShapeNet(x,params,N_l=4,N_s=2,act_fn=F.silu):
    w0 = torch.reshape(params[:,0:N_s],(-1,1,N_s))
    w1 = torch.reshape(params[:,N_s:((N_s+1)*N_s)],(-1,N_s,N_s))
    w2 = torch.reshape(params[:,(N_s**2+N_s):(2*N_s**2+N_s)],(-1,N_s,N_s))
    w3 = torch.reshape(params[:,(2*N_s**2+N_s):(2*N_s**2+2*N_s)],(-1,N_s,1))
    b0   = torch.reshape(params[:,(2*N_s**2+2*N_s):(2*N_s**2+3*N_s)],(-1, N_s))
    b1   = torch.reshape(params[:,(2*N_s**2+3*N_s):(2*N_s**2+4*N_s)],(-1, N_s))
    b2   = torch.reshape(params[:,(2*N_s**2+4*N_s):(2*N_s**2+5*N_s)],(-1, N_s))
    b3   = torch.reshape(params[:,(2*N_s**2+5*N_s):],(-1,1))

    u = act_fn(torch.einsum('ij,ijk->ik',x,w0) + b0)
    u = act_fn(torch.einsum('ij,ijk->ik',u,w1) + b1) + u
    u = act_fn(torch.einsum('ij,ijk->ik',u,w2) + b2) + u
    u = torch.einsum('ij,ijk->ik',u,w3) + b3
    return u