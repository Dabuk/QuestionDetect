required libraries- 
python 3.6, pytorch 1.0.0, jupyter notebook, CoreNLP

data processing- 
use DataPreProcessing.ipynb for preprocessing the data

Extract POStags from the preprocessed data in Conll format 

Run the Conll file through DataFormatPrep.ipynb to create data in the format necessary to train the model


Sample lines in data - 0's are for questions and 1's for statements

0 0 0 0 0 0 0 0 0 0     what kind of request is the weirdest you have gotten    WDT NN IN NN VBZ DT JJS PRP VBP VBN
1 1 1 1 1 1 1 1 1       thats not even true i just want a slice NNP RB RB JJ PRP RB VBP DT NN
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1     oh come on weve seen it im not in the mood to see a fourhour documentary on nazis       UH VBN IN JJ VBN PRP NN RB IN DT NN TO VB DT JJ NN IN NNPS
0 0 0 0 0 0     did they try to escape war      VBD PRP VBP TO VB NN


split the data into training and validation as needed

Training - 
```Python
python -u train.py -data ../data/train_punct.txt -bsz 48 -save ../models/model_joint_punct.pt -lr 0.001 -outunk 2 
```

Inference - Create data in the same format as train/valid file and replace the data in validation file with inference data -  This is because vocabulary is created on the fly using train data. I haven't added the vocab store feature yet. 
```Python
python validate.py -data ../data/train_no_punct.txt -save ../models/model_joint_no_punct.pt -bsz 1 -eval -max 100 -outunk 2
```
The validate.py creates a predictions file in output folder containing 0 / 1 label for each line in the test file. 1 means question, 0 means statement
