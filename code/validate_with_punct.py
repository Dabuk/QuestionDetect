import torch
import argparse
from time import time
from dataset_entity import dataset
from model import model
from pargs import pargs
import torch.nn.functional as F

# predict question = 1 / statement = 0 based on which model maximizes the prediction of '?' as the next word
def test(args):
  ds = dataset(args)
  m,_ = torch.load(args.save)
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

  for b in data:
    # tgt character is '?'
    tgtchar = ds.OUTP.vocab.stoi['?']

    b.q,b.qlen = b.q
    # make b.q 1 , statment model
  
    b.q[0,:b.qlen[0]-1] = ds.REL.vocab.stoi['1']

    # predict '?' under statement model
    b.q = (b.q,b.qlen)
    pq = m(b.tgt,b.pos,b.q)
    pq = F.softmax(pq,dim=-1).log()
    pq = pq[:,:-1,:]
    
    # change q to 0
    b.q,b.qlen = b.q
    b.q[0,:b.qlen[0]-1] = ds.REL.vocab.stoi['0']
    b.q = (b.q,b.qlen)

    # predict '?' under question model
    ps = m(b.tgt,b.pos,b.q)

    ps = F.softmax(ps,dim=-1).log()
    ps = ps[:,:-1,:]

    tgt1 = b.tgt[0][:,:-1].squeeze(0)
    tgt1[tgt1!=1] = tgtchar
    
    # get mean log probability score under each model
    qscore = torch.gather(pq.squeeze(0),-1,tgt1.unsqueeze(1)).mean()
    pscore = torch.gather(ps.squeeze(0),-1,tgt1.unsqueeze(1)).mean()
    
    ss = ['0']
    if pscore > qscore:
       ss = ['1']
    
    # output target sentences
    tgts = ds.reverse(b.tgt[0][:,:])
    for i,s in enumerate(ss):
      jf.write(tgts[i]+'\n')
      pf.write(s+'\n')

if __name__=="__main__":
  args = pargs()
  test(args)
