import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('train_file')
ap.add_argument('test_file')
args = ap.parse_args()


def read_train_data(filename):
    raw_text, labels = [], []
    with open(filename, 'r') as f:
        for line in f:
            if not line:
                continue

            line = line.replace("\n", "").strip()

            label = "B"
            raw_text.append(line.replace(" ", "" ))
            
            for i in range(1,len(line)):
                if i == 0 or line[i-1] == " ":
                     label += "B"
                elif line[i] != " ":
                    label += "I"
            
            labels.append(label)

    return raw_text, labels

def read_test_data(filename):
    raw_text = []

    with open(filename, 'r') as f:
        for line in f:
            if not line:
                continue

            line = line.replace("\n", "").strip()

            raw_text.append(line.replace(" ", "" ))
    
    return raw_text


raw_words, raw_labels = read_train_data(args.train_file)
tst_text = read_test_data(args.test_file)

# Create Vocabulary Mappings (Char to ID, Tag to ID)

# All unique characters
all_chars = sorted(list(set("".join(raw_words))))
char_to_ix = {char: i + 1 for i, char in enumerate(all_chars)} # +1 for padding index
char_to_ix['<PAD>'] = 0 # Padding index
vocab_size = len(char_to_ix)

# All unique tags
tag_to_ix = {"B": 0, "I": 1}
ix_to_tag = {0: "B", 1: "I"}
num_tags = len(tag_to_ix)

print(f"Character Vocabulary Size: {vocab_size}")
print(f"Tag Vocabulary: {tag_to_ix}")

# Custom Dataset Class
class CompoundSegmentationDataset(Dataset):
    def __init__(self, words, labels, char_to_ix, tag_to_ix):
        self.words = words
        self.labels = labels
        self.char_to_ix = char_to_ix
        self.tag_to_ix = tag_to_ix

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        word = self.words[idx]
        tags = self.labels[idx]

        word_indices = [self.char_to_ix[char] for char in word]
        tag_indices = [self.tag_to_ix[tag] for tag in tags]

        return torch.tensor(word_indices, dtype=torch.long), \
               torch.tensor(tag_indices, dtype=torch.long)

# Padding and Batching Function (Collate Function)

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    words, tags = zip(*batch)
    
    lengths = torch.tensor([len(word) for word in words], dtype=torch.long)
    
    padded_words = nn.utils.rnn.pad_sequence(words, batch_first=True, padding_value=char_to_ix['<PAD>'])
    padded_tags = nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-1)

    return padded_words, padded_tags, lengths

# Split Data into Train and Validation sets
X_train, X_test, y_train, y_test = train_test_split(raw_words, raw_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42) 

train_dataset = CompoundSegmentationDataset(X_train, y_train, char_to_ix, tag_to_ix)
val_dataset = CompoundSegmentationDataset(X_val, y_val, char_to_ix, tag_to_ix)
test_dataset = CompoundSegmentationDataset(X_test, y_test, char_to_ix, tag_to_ix)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

# Define the LSTM Model
class LSTMSegmenter(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, num_layers=1, bidirectional=True, dropout=0.5):
        super(LSTMSegmenter, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        self.hidden_dim = hidden_dim
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.hidden2tag = nn.Linear(lstm_output_dim, num_tags)

    def forward(self, sentence, lengths):
        embeds = self.embedding(sentence)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), batch_first=True, enforce_sorted=True)
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        tag_scores = self.hidden2tag(output)
        
        return tag_scores

# Set the hyperparameters
EMBEDDING_DIM = 100
HIDDEN_DIM = 128
NUM_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5
LEARNING_RATE = 0.001
NUM_EPOCHS = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LSTMSegmenter(vocab_size, EMBEDDING_DIM, HIDDEN_DIM, num_tags, NUM_LAYERS, BIDIRECTIONAL, DROPOUT).to(device)
loss_function = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

print(f"\nModel Architecture:\n{model}")
print(f"Using device: {device}")

# Define Training Loop
def train_model(model, train_loader, val_loader, loss_function, optimizer, num_epochs, device):
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for words, tags, lengths in train_loader:
            words, tags = words.to(device), tags.to(device)

            optimizer.zero_grad()
            
            tag_scores = model(words, lengths)
            
            loss = loss_function(tag_scores.view(-1, num_tags), tags.view(-1))
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for words, tags, lengths in val_loader:
                words, tags = words.to(device), tags.to(device)
                
                tag_scores = model(words, lengths)
                loss = loss_function(tag_scores.view(-1, num_tags), tags.view(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_lstm_segmenter.pth")

# Call Training Function
train_model(model, train_loader, val_loader, loss_function, optimizer, NUM_EPOCHS, device)


# Load the best model
model.load_state_dict(torch.load("best_lstm_segmenter.pth"))


# Define a function to make a prediction on a new word
def predict_segments(model, word, char_to_ix, ix_to_tag, device):
    model.eval()
    with torch.no_grad():
        word_indices = [char_to_ix.get(char, char_to_ix['<PAD>']) for char in word] # Handle OOV chars with PAD
        input_tensor = torch.tensor(word_indices, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension
        lengths = torch.tensor([len(word)], dtype=torch.long)
        
        tag_scores = model(input_tensor, lengths)
        predicted_tag_indices = torch.argmax(tag_scores, dim=2).squeeze(0).tolist() # Remove batch dim

        predicted_tags_str = [ix_to_tag[idx] for idx in predicted_tag_indices]

        segments = []
        current_segment_chars = []
        for char, tag in zip(list(word), predicted_tags_str):
            if tag == 'B':
                if current_segment_chars:
                    segments.append("".join(current_segment_chars))
                current_segment_chars = [char]
            else:
                current_segment_chars.append(char)
        if current_segment_chars:
            segments.append("".join(current_segment_chars))
        
        return segments, predicted_tags_str

with open('a3-lstm.predictions', 'wt') as f:
    for word in tst_text:
        predicted_segments, predicted_tags = predict_segments(model, word, char_to_ix, ix_to_tag, device)
        print(f"{' '.join(predicted_segments).strip()}", file=f)

