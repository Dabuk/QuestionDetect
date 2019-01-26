import torch
import argparse
from time import time
from dataset_entity import dataset
from model import model
from pargs import pargs
import torch.nn.functional as F

DEVICE = torch.device('cuda')

def test(args):
  ds = dataset(args)
  m,_ = torch.load(args.save)
  m = m.to(DEVICE)
  m.maxlen = args.max
  m.starttok = ds.OUTP.vocab.stoi['<start>']
  m.endtok = ds.OUTP.vocab.stoi['<eos>']
  m.eostok = ds.OUTP.vocab.stoi['.']
  if args.test:
    print("testing")
    data = ds.test_iter
  else:
    data = ds.val_iter
  model = args.save.split("/")[-1]
  ## files to store outputs
  pf = open("../outputs/"+model+".predictions",'w')
  jf = open("../outputs/"+model+".tgts",'w')
  m.maxlen = args.max
  m.eval()
  
  # maximize luanguage model output under condition that input is question / statement
  for b in data:

    #get language model output under statement condition
    b.q,b.qlen = b.q
    b.q[0,:b.qlen[0]-1] = ds.REL.vocab.stoi['1']
    b.q = (b.q,b.qlen)
    ps = m(b.tgt,b.pos,b.q)
    ps = F.softmax(ps,dim=-1).log()
    ps = ps[:,:-1,:]
    
    #get language model output under question condition
    b.q,b.qlen = b.q
    b.q[0,:b.qlen[0]-1] = ds.REL.vocab.stoi['0']
    b.q = (b.q,b.qlen)
    pq = m(b.tgt,b.pos,b.q)
    pq = F.softmax(pq,dim=-1).log()
    pq = pq[:,:-1,:]
    
    tgt = b.tgt[0][:,1:].contiguous().view(-1)
    
    # get mean log probability score under each model
    qscore = torch.gather(pq.squeeze(0),-1,tgt.unsqueeze(1)).mean()
    pscore = torch.gather(ps.squeeze(0),-1,tgt.unsqueeze(1)).mean()
   
    # predict quesion / statement - my model works on 0 - q ; 1 - s 
    # but only for final testing im prnting the opposite way 
    ss = ['1']
    if pscore > qscore:
       ss = ['0']
   
    # output target sentences
    tgts = ds.reverse(b.tgt[0][:,:])
    for i,s in enumerate(ss):
      jf.write(tgts[i]+'\n')
      pf.write(s+'\n')

if __name__=="__main__":
  args = pargs()
  test(args)
