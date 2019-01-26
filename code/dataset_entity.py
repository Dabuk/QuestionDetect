import  torch
from torchtext import data
import pargs as arg
import itertools
import numpy as np
import csv
import sys
import json

csv.field_size_limit(sys.maxsize)


DEVICE = torch.device('cuda')# 0 if torch.cuda.is_available() else -1

def s2bool(v):
  if v.lower()=='false':
    return False
  else:
    return True

class dataset:

  def __init__(self, args):
    path = args.data

    # build vocab
    self.OUTP = data.Field(sequential=True, batch_first=True, eos_token="<eos>",include_lengths=True)
    self.POS = data.Field(sequential=True, batch_first=True, eos_token="<eos>",include_lengths=True)
    self.REL = data.Field(sequential=True, batch_first=True, eos_token="<eos>",include_lengths=True) 
    fields=[("q",self.REL),("tgt",self.OUTP),("pos",self.POS)]
    train = data.TabularDataset(
              path=args.data, format='tsv',
              fields=fields)
    valid = data.TabularDataset(
              path=args.data.replace("train","val"), format='tsv',
              fields=fields)
    test = data.TabularDataset(
              path=args.data.replace("train","test"), format='tsv',
              fields=fields)

    self.OUTP.build_vocab(train, min_freq=args.outunk)
    self.POS.build_vocab(train, min_freq=args.outunk)
    self.REL.build_vocab(train, min_freq=args.outunk)


    self.trainlen = len(train)

    # create iterator
    self.train_iter = data.BucketIterator(train,args.bsz,device=DEVICE, sort_key=lambda x:len(x.tgt), repeat=False, shuffle=True, sort=False)#,sort_within_batch=True)

    if args.eval:
      print("evaluating, no sort")
      self.val_iter = data.BucketIterator(valid,args.vbsz,device=DEVICE,sort_key=lambda x:len(x.tgt), train=False, sort=False)
      self.test_iter = data.BucketIterator(test,args.vbsz,device=DEVICE,sort_key=lambda x:len(x.tgt), train=False, sort=False)
    else:
      self.val_iter = data.BucketIterator(valid,args.vbsz,device=DEVICE,sort_key=lambda x:len(x.tgt),train=False, sort=False)
      self.test_iter = data.BucketIterator(test,args.vbsz,device=DEVICE,sort_key=lambda x:len(x.tgt),train=False, sort=False)
 
  # generate sentences
  def reverse(self,preds):
    vocab = self.OUTP.vocab
    ss = []
    for i,x in enumerate(preds):
      s = ' '.join([vocab.itos[y] for j,y in enumerate(x)])
      if "<eos>" in s: 
        s = s.split("<eos>")[0]
      ss.append(s)
    if len(ss[0].split(' ')) == 0:
        return ['.']
    return ss

  # pad sequence
  def pad(self,tensor, length,ent=1):
    return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).fill_(ent)])
