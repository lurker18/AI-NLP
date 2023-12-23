import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizerFast, RobertaConfig, RobertaForTokenClassification
from torch import cuda

device = 'cuda' if cuda.is_available() else 'cpu'

data_folder = './Dataset/Scraped/nhsinform/'

df = pd.read_csv(data_folder + 'nhsinform-IOB - Modified - Copy.txt', 
                 sep = '\t', 
                 header = None,
                 names = ['Word', 'Annotation']
                 )

df['number'] = 0
sentence_number = 1
for index, row in df.iterrows():
    if row['Word'] == '###':
        df.at[index, 'number'] = sentence_number
        sentence_number += 1
    else:
        df.at[index, 'number'] = sentence_number

df = df[~df['Annotation'].isin(['###'])]

# Add a temporary 'sentence #' column for creating a joined words of sentences
df['Sentence #'] = 'TEMP'
labels_to_ids = {k : v for v, k in enumerate(df.Annotation.unique())}
ids_to_labels = {v : k for v, k in enumerate(df.Annotation.unique())}

df['Word'] = df['Word'].astype('string')
df['Annotation'] = df['Annotation'].astype('string')

#index = df['Annotation'].index[df['Annotation'].apply(pd.isnull)]

# Create a new column called sentence which groups the words by sentence
df['sentence'] = df[['Sentence #', 'Word', 'Annotation', 'number']].groupby(['number'])['Word'].transform(lambda x : ' '.join(x))
# Also, create a new column called 'word_labels' which groups the tags by sentence
df['word_labels'] = df[['Sentence #', 'Word', 'Annotation', 'number']].groupby(['number'])['Annotation'].transform(lambda x : ','.join(x))

df = df[['sentence', 'word_labels']].drop_duplicates().reset_index(drop = True)

df['sentence'] = df['sentence'].apply(lambda x: x.replace(' ###', ''))
df['word_labels'] = df['word_labels'].apply(lambda x: x.replace(',###', ''))

#df.to_json(data_folder + 'NHSInform-IOB-JSON.json', orient = 'records', lines = True)
df.head()

# Test a sample of how the sentence and the word_labels are structured
df.iloc[41].sentence
df.iloc[41].word_labels

# Hyperparameter-Settings
max_len = 512
train_bs = 64
valid_bs = 32
epochs = 10
lr = 1e-05
max_grad_norm = 10
tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")

# Create the dataset class
class dataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

  def __getitem__(self, index):
        # Stage 1: Load the sentence and word labels
        sentence = self.data['sentence'][index]
        word_labels = self.data['word_labels'][index].split(",")

        # Stage 2: Tokenize and encode each sentences
        encoding = self.tokenizer(sentence,
                                  return_offsets_mapping = True, 
                                  padding = 'max_length', 
                                  truncation = True, 
                                  max_length = self.max_len)
        
        # Stage 3: Create token labels only for first word pieces of each tokenized word
        labels = [labels_to_ids[label] for label in word_labels] 
        encoded_labels = np.ones(len(encoding["offset_mapping"]), dtype = int) * -100
        i = 0
        for idx, mapping in enumerate(encoding["offset_mapping"]):
            if mapping[0] != 0 and mapping[0] != encoding['offset_mapping'][idx-1][1]:
                try:
                    encoded_labels[idx] = labels[i]
                except:
                    pass
                i += 1
            else:
                if idx==1:
                    encoded_labels[idx] = labels[i]
                    i += 1
        # step 4: turn everything into PyTorch tensors
        item = {key: torch.as_tensor(val) for key, val in encoding.items()}
        item['labels'] = torch.as_tensor(encoded_labels)
        
        return item

  def __len__(self):
        return self.len


# split train / test sets
ts = 0.8
train_dataset = df.sample(frac = ts, random_state = 42)
test_dataset = df.drop(train_dataset.index).reset_index(drop = True)
train_dataset = train_dataset.reset_index(drop = True)

print("Full Dataset: {}".format(df.shape))
print("Train Dataset: {}".format(train_dataset.shape))
print("Test Dataset: {}".format(test_dataset.shape))

training_set = dataset(train_dataset, tokenizer, max_len)
testing_set = dataset(test_dataset, tokenizer, max_len)

# Load PyTorch DataLoader
train_params = {'batch_size': train_bs,
                'shuffle' : True,
                'pin_memory': True}

test_params = {'batch_size': valid_bs,
                'shuffle' : True,
                'pin_memory' : True}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

model = RobertaForTokenClassification.from_pretrained('roberta-base',
                                                       num_labels = len(labels_to_ids))
model.to(device)

# Test run a sample
#inputs = training_set[0]
#input_ids = inputs["input_ids"].unsqueeze(0)
#attention_mask = inputs["attention_mask"].unsqueeze(0)
#labels = inputs['labels'].unsqueeze(0)

#input_ids = input_ids.to(device)
#attention_mask = attention_mask.to(device)
#labels = labels.type(torch.LongTensor)
#labels = labels.to(device)

#outputs = model(input_ids, 
#                attention_mask = attention_mask,
#                labels = labels,
#                return_dict = False)

#initial_loss = outputs[0]
#initial_loss

# Setup the optimizer
optimizer = torch.optim.Adam(params = model.parameters(), 
                             lr = lr)

def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    
    # Train the model
    model.train()
    
    for idx, batch in enumerate(training_loader):
        
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        labels = batch['labels'].to(device, dtype = torch.long)

        loss, tr_logits = model(input_ids = ids, 
                                attention_mask = mask, 
                                labels = labels,
                                return_dict = False)
        tr_loss += loss.item()

        nb_tr_steps += 1
        nb_tr_examples += labels.size(0)
        
        if idx % 100 == 0:
            loss_step = tr_loss/nb_tr_steps
            print(f"Training loss per 100 training steps: {loss_step}")
           
        # compute training accuracy
        flattened_targets = labels.view(-1)                         # shape: (batch_size * seq_len,)
        active_logits = tr_logits.view(-1, model.num_labels)        # shape: (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape: (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100                   # shape: (batch_size, seq_len)
        
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        tr_labels.extend(labels)
        tr_preds.extend(predictions)

        tmp_tr_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
        tr_accuracy += tmp_tr_accuracy
    
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(
            parameters = model.parameters(), 
            max_norm = max_grad_norm
        )
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    epoch_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {epoch_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")

def valid(model, testing_loader):
    # put model in evaluation mode
    model.eval()
    
    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []
    
    with torch.no_grad():
        for idx, batch in enumerate(testing_loader):
            
            ids = batch['input_ids'].to(device, dtype = torch.long)
            mask = batch['attention_mask'].to(device, dtype = torch.long)
            labels = batch['labels'].to(device, dtype = torch.long)
            
            loss, eval_logits = model(input_ids = ids, 
                                      attention_mask = mask, 
                                      labels = labels,
                                      return_dict = False)
            
            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += labels.size(0)
        
            if idx % 100 == 0:
                loss_step = eval_loss/nb_eval_steps
                print(f"Validation loss per 100 evaluation steps: {loss_step}")
              
            # compute evaluation accuracy
            flattened_targets = labels.view(-1)                           # shape (batch_size * seq_len,)
            active_logits = eval_logits.view(-1, model.num_labels)        # shape (batch_size * seq_len, num_labels)
            flattened_predictions = torch.argmax(active_logits, axis = 1) # shape (batch_size * seq_len,)
            
            # only compute accuracy at active labels
            active_accuracy = labels.view(-1) != -100                     # shape (batch_size, seq_len)
        
            labels = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)
            
            eval_labels.extend(labels)
            eval_preds.extend(predictions)
            
            tmp_eval_accuracy = accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())
            eval_accuracy += tmp_eval_accuracy

    labels = [ids_to_labels[id.item()] for id in eval_labels]
    predictions = [ids_to_labels[id.item()] for id in eval_preds]
    
    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    return labels, predictions

# Train the model with # of epochs
for epoch in range(epochs):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)

# Inference Phase
sentence = "Anaphylaxis is a severe, potentially life-threatening allergic reaction that can develop rapidly. It is also known as anaphylactic shock."
model.eval()
def inference(sentence):
    inputs = tokenizer(sentence,
                       return_offsets_mapping = True, 
                       padding = 'max_length', 
                       truncation = True, 
                       max_length = max_len,
                       return_tensors = "pt")

    # move to gpu
    ids = inputs["input_ids"].to(device)
    mask = inputs["attention_mask"].to(device)
    # forward pass
    outputs = model(ids, 
                    attention_mask = mask, 
                    return_dict = False)
    logits = outputs[0]

    active_logits = logits.view(-1, model.num_labels)             # shape (batch_size * seq_len, num_labels)
    flattened_predictions = torch.argmax(active_logits, axis = 1) # shape (batch_size*seq_len,) - predictions at the token level

    tokens = tokenizer.convert_ids_to_tokens(ids.squeeze().tolist())
    token_predictions = [ids_to_labels[i] for i in flattened_predictions.cpu().numpy()]
    wp_preds = list(zip(tokens, token_predictions)) # list of tuples. Each tuple = (wordpiece, prediction)

    prediction = []
    out_str = []
    off_list = inputs["offset_mapping"].squeeze().tolist()
    for idx, mapping in enumerate(off_list):

        if mapping[0] != 0 and mapping[0] != off_list[idx-1][1]:
            prediction.append(wp_preds[idx][1])
            out_str.append(wp_preds[idx][0])
        else:
            if idx == 1:
                prediction.append(wp_preds[idx][1])
                out_str.append(wp_preds[idx][0])
            continue
    return prediction, out_str

y_pred = []

for i, t in enumerate(test_texts['sentence'].tolist()):
    o,o_t = inference(t)
    y_pred.append(o)
    l = df['word_labels'][i]