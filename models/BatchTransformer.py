import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
import math
import torch.nn as nn
from torch import Tensor

class ChannelPositionalEncoding(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, channel))
        #self.channel = channel
        
    def forward(self, q, k):
        B, C, _, _ = q.shape
        q = q.view(B, C, -1).permute(0, 2, 1)  # BCHW -> BNC
        q = q + self.pos_embed # B,E,C + B,1,C . Broadcasting So E dimension has same position 
        q = q.contiguous().transpose(-2,-1) #B,C,E

        k = k.view(B, C, -1).permute(0, 2, 1)  # BCHW -> BNC
        k = k + self.pos_embed # B,E,C
        k = k.contiguous().transpose(-2,-1) #B,C,E
        return q, k

class BatchTransformer(nn.Module):
    def __init__(self,
                 channel,
                 ):
        factory_kwargs = {'device': None, 'dtype': None}
        super().__init__()
        self.pos_encoder = ChannelPositionalEncoding(channel=channel)
        
    def forward(self, query,key,value):

        q,k = self.pos_encoder(query,key) #B,C,49
        output_CBA = ClassBatchAttention(q,k,value)
        output_BT = output_CBA + value

        return output_CBA, output_BT
    
def ClassBatchAttention(q,k,value):

    #query :B,C,E -> 1,C,B,E
    q = q.contiguous().transpose(0,1).unsqueeze(0) #1, C, B, 49
    k = k.contiguous().transpose(0,1).unsqueeze(0)

    #value :B,C -> C,B,1
    v = value.unsqueeze(1) #B, 1, C
    v = v.contiguous().transpose(0,-1)#C,1,B
    v = v.contiguous().transpose(-2,-1)#C,B,1

    _,C,B,_ = q.shape #1, C, B, 49
    q_scaled = q / math.sqrt(C)
    attn_output_weights = q_scaled @ k.transpose(-2, -1)#1,C,B,B
    attn_output_weights = torch.softmax(attn_output_weights, dim=-1) #1,C,B,B
    attn_output_weights = attn_output_weights.contiguous().view(C,B,B) #C,B,B

    attn_output = torch.bmm(attn_output_weights, v) #C,B,1
    attn_output = attn_output.contiguous().view(C, B)#C,B
    attn_output = attn_output.transpose(-1,-2) #B,C
    return attn_output