import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from beam import Beam
from attention import MatrixAttn

#DEVICE = 0 if torch.cuda.is_available() else -1
DEVICE = torch.device('cuda')

class model(nn.Module):

  def __init__(self,args):
    super(model, self).__init__()
    outtoks, embsize, hsize = args.ntoks, args.esz, args.hsz
    self.outemb = nn.Embedding(outtoks,args.esz)
    self.posemb = nn.Embedding(args.posembs,args.esz) 
    self.qemb = nn.Embedding(args.rtoks,args.edgesz)
    self.vtoks = args.vtoks
    self.encoder = nn.LSTM(args.esz*2 + args.edgesz , hsize//2, batch_first=True, num_layers=2, bidirectional=True, dropout=args.drop)
    self.outlin = nn.Linear(hsize*2, outtoks)
    self.entemb_drop = nn.Dropout(args.drop)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.maxlen = args.max
    self.outtoks = outtoks
    self.endtok = args.endtok
    self.edgesz = args.edgesz
    self.embsize = embsize
    self.hsize = hsize
    self.matattn = MatrixAttn(hsize,hsize)

  # bidirectional, self attention based model
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
    
    e, (h,c) = self.encoder(decin)    
   
    out, attn = self.matattn(e,(e,outlen))
    e = torch.cat([e,out],dim=-1)
   
    o = self.outlin(e)

    return o
   
  def forward(self,outp,pos,q):

    o = self.langModel(outp,pos,q)
    
    return o 
