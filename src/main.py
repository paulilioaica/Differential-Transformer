from datasets import load_dataset
from pretrained_tokenizer import tokenizer
from transformer import Transformer
from dataset import TextDataset
from torch.utils.data import DataLoader
from trainer import run_trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--n_heads", type=int, default=8)
parser.add_argument("--seq_len", type=int, default=128)
parser.add_argument("--num_epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)   
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--device", type=str, default="cpu")
parser.add_argument("--embedding_dim", type=int, default=128)
parser.add_argument("--dataset_name", type=str, default="ag_news")
args = parser.parse_args()



def run(num_layers, n_heads, seq_len, num_hidden, num_epochs, batch_size, lr, device, embedding_dim, dataset_name):
    # add split="train[10%:20%]" to load_dataset to get a smaller dataset
    dataset_train = load_dataset(dataset_name, split='train[:1%]')
    dataset_test = load_dataset(dataset_name, split='test[:2%]') 

    train_dataset = dataset_train
    test_dataset = dataset_test


    tokenized_train_dataset = train_dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_len), 
                                                batched=True)
    tokenized_test_dataset = test_dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=seq_len), 
                                              batched=True)


    train_dataset = TextDataset(tokenized_train_dataset)
    test_dataset = TextDataset(tokenized_test_dataset)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model = Transformer(num_layers, num_layers, num_hidden, n_heads, seq_len - 1, tokenizer.vocab_size, embedding_dim)
    
    results = run_trainer(model, train_dataloader, test_dataloader, num_epochs, device, lr)
    return results

if __name__ == "__main__":
    run(args.num_layers, args.n_heads, args.seq_len, args.embedding_dim, args.num_epochs, args.batch_size, args.lr, args.device, args.embedding_dim, args.dataset_name)
