#!/usr/bin/env python
# coding: utf-8

# # Text Classification

# ## DataSetProcess

# In[1]:


#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding: UTF-8
import torch
import numpy as np
from importlib import import_module
import pandas as pd
import pandas as pd
import numpy as np
from torchtext.vocab import GloVe
import torch.nn as nn
from sklearn.metrics import accuracy_score
glove = GloVe(name='6B', dim=300)
import nltk
from string import punctuation
def cut_word(sentence):
    tokens = nltk.word_tokenize(sentence)
    tokens_without_punct = [word.lower() for word in tokens if word not in punctuation]
    return ' '.join(tokens_without_punct)  

def sentence_to_vector(word_list):
    tensor = glove.get_vecs_by_tokens(word_list.split(' '), True).numpy().tolist()
    for i in range(100-len(tensor)):
        tensor.append([0]*300)
    return tensor


# In[2]:


train_data = pd.read_csv('./dataset/train.csv')
test_data = pd.read_csv('./dataset/test.csv')
dev_data = pd.read_csv('./dataset/dev.csv')
rawdata = pd.concat([train_data,test_data,dev_data])
rawdata.index = range(len(rawdata))
rawdata


# ### Make Datasets

# In[3]:


train_label_index=np.array(list(range(len(train_data))))
dev_label_index=np.array(list(range(len(train_data),len(train_data)+len(dev_data))))
test_label_index=np.array(list(range(len(train_data)+len(dev_data),len(train_data)+len(dev_data)+len(test_data))))
np.random.shuffle(train_label_index)
np.random.shuffle(dev_label_index)


# In[4]:


data=rawdata[['label','text']]
data['text']=data['text'].apply(lambda x:x[:100]) # seq_length
data.columns=['labels','texts']
data.head()


# In[5]:


data['word_list']=data.texts.apply(cut_word)
data


# ### Glove embedding pre training word encoding maps each word to a vector of 300

# ### Pytorch DataLoader

# In[6]:


import torch.utils.data as Data
class MyDataSet(Data.Dataset):
    def __init__(self, data):
        self.x = data['word_list'].values
        self.y = data.labels.values

    def __getitem__(self, index):
        return np.array(sentence_to_vector(self.x[index]),dtype=np.float32),self.y[index]

    def __len__(self):
        return len(self.x)
dataset = MyDataSet(data)

batch_size=100
train_iter = Data.DataLoader(dataset, batch_size=batch_size,shuffle=False, drop_last=False, sampler=train_label_index)
dev_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False,drop_last=False, sampler=dev_label_index)
test_iter = Data.DataLoader(dataset, batch_size=3000, shuffle=False,drop_last=False, sampler=test_label_index)

model_name ='TextRNN_Att' 


# In[7]:


# Set model parameters
x = import_module(f'models.{model_name}')
config = x.Config()
config.num_classes = 4
config.learning_rate = 0.001
config.num_epochs = 30

# Load Model
model = x.Model(config).to(config.device)

# Set optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
loss_func = nn.CrossEntropyLoss()


# In[8]:


# Record losses and accuracy during training
train_loss_list = []
train_acc_list = []
test_loss_list = []
test_acc_list = []

train_loss_record = []
train_acc_record = []
test_loss_record = []
test_acc_record = []

# Training
for epoch in range(config.num_epochs):
    print(f'Epoch [{epoch + 1}/{config.num_epochs}]')
    model.train()  # 
    for i, (inputs, targets) in enumerate(train_iter):
        inputs, targets = inputs.to(config.device), targets.to(config.device)
        optimizer.zero_grad()  
        outputs = model(inputs)  
        loss = loss_func(outputs, targets) 
        loss.backward()  
        optimizer.step()  
        train_preds = torch.argmax(outputs, axis=1)
        train_acc = accuracy_score(train_preds.cpu(), targets.cpu())
        train_loss_list.append(loss.item())
        train_acc_list.append(train_acc)
# Eval
    model.eval()  
    with torch.no_grad():
        test_loss_total = 0
        test_acc_total = 0
        test_count = 0
        for inputs, targets in dev_iter:
            inputs, targets = inputs.to(config.device), targets.to(config.device)
            outputs = model(inputs)
            test_loss = loss_func(outputs, targets).item()
            test_preds = torch.argmax(outputs, axis=1)
            test_acc = accuracy_score(test_preds.cpu(), targets.cpu())
            test_loss_total += test_loss
            test_acc_total += test_acc
            test_count += 1
        test_loss_avg = test_loss_total / test_count
        test_acc_avg = test_acc_total / test_count
        test_loss_list.append(test_loss_avg)
        test_acc_list.append(test_acc_avg)

    train_loss_record.append(train_loss_list[-1])
    train_acc_record.append(train_acc_list[-1])
    test_loss_record.append(test_loss_list[-1])
    test_acc_record.append(test_acc_list[-1])

    print(f'Training Loss: {train_loss_list[-1]:.4f}, Training Acc: {train_acc_list[-1]:.4f}',f'Evaling Loss: {test_loss_list[-1]:.4f}, Evaling Acc: {test_acc_list[-1]:.4f}')


# In[9]:


# Save the losses and accuracy during training to DataFrame
records = pd.DataFrame({
    'Train Loss': train_loss_record,
    'Train Acc': train_acc_record,
    'Eval Loss': test_loss_record,
    'Eval Acc': test_acc_record
})


# In[10]:


records


# In[25]:


import matplotlib.pyplot as plt

# 设置图形尺寸
plt.plot(records['Train Acc'], label='Training Accuracy')
plt.plot(records['Eval Acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epoch')
plt.grid()
plt.savefig('Accuracy.png')
plt.show()
plt.clf()
plt.plot(records['Train Loss'], label='Training Loss')
plt.plot(records['Eval Loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.title('Loss over Epoch')
plt.savefig('Loss.png')

# 显示图形
plt.show()


# In[14]:


with torch.no_grad():
    for tests, labels in test_iter:
        tests, labels = tests.cuda().float(), labels.long().cuda()
        outputs = model(tests)  # 10,73->10,5
        test_preds = torch.argmax(outputs, axis=1)


# In[15]:


groundtruth, predict = labels.cpu().numpy(), test_preds.cpu().numpy()


# In[16]:


import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
label_column = ['Low Institutional Trust', 'Low Agent Trust', 'Neutral', 'High Trust']
report=classification_report(groundtruth, predict,target_names=label_column,output_dict=True)
report = pd.DataFrame(report)
report.to_csv('./result.csv')
report


# In[26]:


plt.figure(figsize=(6,4))
report.T.iloc[:4,:3].T.plot.bar(rot=0,ax=plt.gca())
plt.legend(fontsize=12,bbox_to_anchor=(0.90,-0.10),ncol=2)
plt.xticks(size=15)
plt.savefig('Compare.png')
plt.show()


# In[27]:


cm = confusion_matrix(groundtruth, predict)#
ax = sns.heatmap(cm,annot=True,fmt='g',xticklabels=label_column,yticklabels=label_column,annot_kws={"fontsize":20})
#xticklabels、yticklabels
ax.set_xlabel('Predict',size=20) #x
ax.set_ylabel('GroundTruth',size=20) #y
plt.xticks(fontsize=10) 
plt.yticks(fontsize=10) 
plt.gcf().set_size_inches(8, 6)
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=15)
plt.savefig('Confusion_Matrix.png')
plt.show() 


# In[ ]:




