import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

DEVICE = torch.device('cuda')

class model(nn.Module):
  # initialize model
  def __init__(self,args):
    super(model, self).__init__()
    outtoks, embsize, hsize = args.ntoks, args.esz, args.hsz
    self.outemb = nn.Embedding(outtoks,args.esz)
    self.posemb = nn.Embedding(args.posembs,args.esz) 
    self.qemb = nn.Embedding(args.rtoks,args.edgesz)
    self.vtoks = args.vtoks
    self.dlstm = nn.LSTMCell(args.esz*2 + args.edgesz, hsize)
    self.outlin = nn.Linear(hsize, outtoks)
    self.entemb_drop = nn.Dropout(args.drop)
    
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.maxlen = args.max
    self.outtoks = outtoks
    self.endtok = args.endtok
    self.edgesz = args.edgesz
    self.embsize = embsize
    self.hsize = hsize
 
  # lstm decoder 
  def decode_single(self, k, h, c):
    h, c = self.dlstm(k,(h,c))
    decoded = self.outlin(h.contiguous())
    return decoded, h, c
  # language model
  def langModel(self, outp, pos, q, h=None, c=None):

    outp, outlen = outp
    pos, poslen = pos
    q, qlen = q
    
    outp = self.outemb(outp)
    outp = self.entemb_drop(outp)
    pos = self.posemb(pos)
    pos = self.entemb_drop(pos)
  
    q = self.qemb(q)
    q = self.entemb_drop(q)
    
    decin = torch.cat([outp,pos,q],dim=-1)    
    decin = decin.transpose(0,1)

    h  = torch.zeros(outp.size(0), self.hsize).to(DEVICE)
    c = torch.zeros(outp.size(0), self.hsize).to(DEVICE)
    
    outputs = []
    for k in decin:
        o, h, c = self.decode_single(k, h, c)
        outputs.append(o)    
    o = torch.stack(outputs,1) 
    return o
   
  def forward(self,outp,pos,q):

    o = self.langModel(outp,pos,q)
    
    return o 
