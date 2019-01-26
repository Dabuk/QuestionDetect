import torch
import math
from time import time
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from dataset_entity import dataset
from model import model
from pargs import pargs
from torchtext import data

DEVICE = torch.device('cuda')

def main(args):
  ds = dataset(args)
  
  # define number of tokens in tgt, postags and acts (question / statement)
  args.vtoks = args.ntoks = len(ds.OUTP.vocab.itos)
  args.posembs = len(ds.POS.vocab.itos)
  args.rtoks = len(ds.REL.vocab.itos)
  args.endtok = ds.OUTP.vocab.stoi["<eos>"]
  print("building model")

  m = model(args)
  c = nn.CrossEntropyLoss(ignore_index=1)
  if torch.cuda.is_available():
    m = m.to(DEVICE)
    c = c.to(DEVICE)
  print("built model")
  print(m)
  o = torch.optim.Adam(m.parameters(),lr=args.lr,weight_decay=1e-8)#, momentum=0.9)

  # training
  oldvloss = 1000000
  bestvloss = 1000000
  oldvacc = 0
  bestvacc = 0
  stopcount = 0
  for epoch in range(args.epochs):
    t = time()
    print("EPOCH ",epoch)
    losses = 0
    examples = 0
    for b in ds.train_iter:
      o.zero_grad()
      # limit max len for BPTT
      oup,_ = b.tgt
      q,_ = b.q
      pos,_ = b.pos
      if oup.size(1) > 100:
         oup = oup[:,:100]
         q = q[:,:100]
         pos = pos[:,:100]
      b.tgt = (oup,b.tgt[1])
      b.q = (q,b.q[1])
      b.pos = (pos,b.pos[1])
     
      p = m(b.tgt,b.pos,b.q)
      p = p[:,:-1,:]
      tgt = b.tgt[0][:,1:].contiguous().view(-1)
      tempmax,_ = torch.max(tgt,dim=-1)
      l = c(p.contiguous().view(-1,p.size()[-1]),tgt) # + 0.1*closs
      l.backward()
      # calculate loss for printing
      losses += l.item() * len(b.tgt)
      examples += len(b.tgt)
      torch.nn.utils.clip_grad_norm_(m.parameters(), 50)
      o.step()
      
      if math.isnan(losses):
         print("loss contains nan")
         exit()
    losses = losses/examples
    print("AVG TRAIN LOSS: ",losses)
    if losses < 100: print(" PPL: ",math.exp(losses))
    print("time :",time()-t)

    # evaluate 
    m.eval()
    vloss = 0
    right = 0
    guesses = 0
    examples = 0
    for b in ds.val_iter:
      _,outlens = b.tgt
      # check if null string is passed
      if b.tgt[0].size()[1] <=1:
        continue
        
      oup,_ = b.tgt
      q,_ = b.q
      pos,_ = b.pos
      if oup.size(1) > 100:
         oup = oup[:,:100]
         q = q[:,:100]
         pos = pos[:,:100]
      b.tgt = (oup,b.tgt[1])
      b.q = (q,b.q[1])
      b.pos = (pos,b.pos[1])

      p = m(b.tgt,b.pos,b.q)
      p = p[:,:-1,:]
      tgt = b.tgt[0][:,1:].contiguous().view(-1)
      l = c(p.contiguous().view(-1,p.size()[-1]),tgt) #+ 0.1*closs
      _, maxes = p.max(2)
      outlen = len(tgt)-tgt.eq(1).sum().item()
    
      # get accuracy of predicted words
      acc = tgt.view_as(maxes).eq(maxes)
      acc[(acc.long() >= outlens.unsqueeze(1))].fill_(0)
      acc = acc.sum().item()
      right += acc
      guesses += outlen
      vloss += float(l) * len(b.tgt[0])
      examples += len(b.tgt[0])
    m.train()
    vloss = vloss/examples
    vacc = right/guesses
    print("VAL LOSS: ",vloss)
    if vloss < 100: print(" PPL: ",math.exp(vloss))
    print("VAL ACC: ",vacc, right, guesses, examples)
    if vloss <= bestvloss:
      torch.save((m,o),args.save)
      print("Saving model")
      bestvloss = vloss
    if vloss > oldvloss:
      stopcount+=1
      if stopcount >= 6:
         print("no decrease in loss. Exiting")
         torch.save((m,o),args.save+'last_epoch')
         return
      o.param_groups[0]['lr'] *= 0.5
      print("decay lr to ",o.param_groups[0]['lr'])
    oldvloss = vloss

if __name__=="__main__":
  args = pargs()
  if args.eval:
    test(args)
  else:
    main(args)
