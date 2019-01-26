import argparse

def pargs():
  parser = argparse.ArgumentParser(description='Graph Doc Plan')

  #model
  parser.add_argument("-model",default="default")
  parser.add_argument("-edgesz",default=100,type=int)
  parser.add_argument("-esz",default=300,type=int)
  parser.add_argument("-hsz",default=500,type=int)
  parser.add_argument("-drop",default=0.1,type=float)

  # training and loss
  parser.add_argument("-bsz",default=20,type=int)
  parser.add_argument("-vbsz",default=1,type=int)
  parser.add_argument("-epochs",default=50,type=int)
  parser.add_argument("-clip",default=1,type=float)

  #optim
  parser.add_argument('-b1', type=float, default=0.9)
  parser.add_argument('-b2', type=float, default=0.999)
  parser.add_argument("-lr",default=0.0006,type=float)
  parser.add_argument('-max_grad_norm', type=int, default=1)

  #data
  parser.add_argument("-nosave",action='store_false')
  parser.add_argument("-save",default="../models/model.pt")
  parser.add_argument("-outunk",default=0,type=int)
  parser.add_argument("-data",default="../data/amr_train_data.txt")
  #eval
  parser.add_argument("-eval",action='store_true')

  #inference
  parser.add_argument("-max",default=200,type=int)
  parser.add_argument("-test",action='store_true')
  parser.add_argument("-sample",action='store_true')
  args = parser.parse_args()

  return args
