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
    self.vtoks = args.vtoks
    self.encoder = nn.LSTM(args.esz*2,hsize//2,batch_first=True,num_layers=2,bidirectional=True,dropout=args.drop)
    self.outlin = nn.Linear(hsize*2, hsize*2)
    self.entemb_drop = nn.Dropout(args.drop)
    self.out = nn.Linear(hsize*2,2)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()
    self.maxlen = args.max
    self.outtoks = outtoks
    self.edgesz = args.edgesz
    self.embsize = embsize
    self.hsize = hsize

    self.matattn = MatrixAttn(hsize,hsize)

  def _cat_directions(self, h):
    h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
    return h

  # encode entitiy
  def encode(self,inp):
    inp, ilens = inp
    #hsize = self.hsize
    #e = pack_padded_sequence(e,ilens,batch_first=True)
    e, (h,c) = self.encoder(inp)
    #h = torch.cat([h[i] for i in range(h.size(0))], 1).unsqueeze(1)
    h = self._cat_directions(h)
    c = self._cat_directions(c)
    #e = pad_packed_sequence(e,batch_first=True)
    return e, (h,c)

  # self attn dialogue act model
  def diaModel(self, outp, pos, h=None, c=None):

    outp, outlen = outp
    pos, poslen = pos
    
    outp = self.outemb(outp)
    outp = self.entemb_drop(outp)
    pos = self.posemb(pos)
    pos = self.entemb_drop(pos)
  
    decin = torch.cat([outp,pos],dim=-1)    
   
    e,(h,_) = self.encode((decin,outlen))

    out, attn = self.matattn(e,(e,outlen))

    e = torch.cat([e,out],dim=-1)
     
    e = torch.mean(e,dim=1)
    return e
   
  def forward(self,outp,pos):

    o = self.langModel(outp,pos)
    o = self.tanh(self.outlin(o))
    o = self.out(e) 
    return o #torch.cat((decoded,attnents),dim=-1).log()
