# Versions:
# Python 3.8.12
# PyTorch 1.13.1+cu117
# Numpy 1.21.5

import argparse
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from cnn_architecture import CNN
from t5 import get_ProtT5_UniRef50_embedding

#### Padding, batching and loader functions ####
def pad_and_batch_sequences(embeddings_dict, size, batch_size):
    uniprots = list(embeddings_dict.keys())
    X = list(embeddings_dict.values())

    mask = [np.ones(elem.shape[0], dtype=int) for elem in X]
    tensor_sequences_X = [torch.tensor(seq, dtype=torch.float) for seq in X]
    tensor_sequences_mask = [torch.tensor(seq, dtype=torch.float) for seq in mask]

    padded_sequences_X = pad_sequence(tensor_sequences_X, batch_first=True, padding_value=0.0)
    padded_sequences_mask = pad_sequence(tensor_sequences_mask, batch_first=True, padding_value=0.0)

    total_sequences_X = padded_sequences_X.shape[0]
    X_batches = [padded_sequences_X[i:i + batch_size] for i in range(0, total_sequences_X, batch_size)]

    total_sequences_mask = padded_sequences_mask.shape[0]
    mask_batches = [padded_sequences_mask[i:i + batch_size] for i in range(0, total_sequences_mask, batch_size)]
    uniprots_batches = [uniprots[i:i + batch_size] for i in range(0, total_sequences_mask, batch_size)]

    return uniprots_batches, X_batches, mask_batches

def making_loader(embeddings_dict, size, batch_size):
    uniprots_batches, X_batches, mask_batches = pad_and_batch_sequences(embeddings_dict, size, batch_size)
    loader_set = [(X, mask) for X, mask in zip(X_batches, mask_batches)]
    return uniprots_batches, loader_set

#### Argument parser ####
def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', help='Path to the input FASTA file', required=True)
    parser.add_argument('--embedding_mode', help='Load or compute embeddings', choices=['load', 'compute'], required=True)
    parser.add_argument('--embed_dir', help='Directory containing precomputed embeddings (.npy files)')
    parser.add_argument('--prott5_model_dir', help='Directory of ProtT5 model (if compute mode)')
    return parser.parse_args()

#### Main function ####
def main():
    args = parser()

    # Set random seeds
    random_seed = 4
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Model parameters
    lr = 1e-5
    weight_decay = 1e-3
    channels = 512
    dropout = 0.75
    sequence_cut = 5000
    # batch_size=1: la conv depthwise (groups=512) con dim spaziale (N,1) e batch>1
    # crasha su CPU/oneDNN ("vector too long"). Con batch=1 il forward e' corretto e
    # i risultati sono identici (conv zero-padded + BatchNorm in eval => indipendenti dal batch).
    batch_size = 1

    # Read input FASTA
    input_split = {}
    with open(args.input_file, "r") as f:
        lines = f.read().splitlines()
    ind = 0
    while ind < len(lines):
        input_split[lines[ind][1:]] = lines[ind + 1]  # Skip '>' for protein ID
        ind += 2

    # Load or compute embeddings
    embeddings_dict = {}
    if args.embedding_mode == 'load':
        if not args.embed_dir:
            raise ValueError("embed_dir must be provided in 'load' mode")
        for prot_id in input_split.keys():
            npy_file = os.path.join(args.embed_dir, f'{prot_id}.npy')
            if os.path.exists(npy_file):
                embedding = np.load(npy_file)  # shape (L, 1024)
                embeddings_dict[prot_id] = embedding
            else:
                print(f"Warning: embedding file for {prot_id} not found in {args.embed_dir}")

    elif args.embedding_mode == 'compute':
        embeddings_dict = get_ProtT5_UniRef50_embedding(
            fasta_path=args.input_file,
            model_dir=args.prott5_model_dir)
        
        if args.embed_dir:
            os.makedirs(args.embed_dir, exist_ok=True)
            for prot_id, embedding in embeddings_dict.items():
                np.save(os.path.join(args.embed_dir, f"{prot_id}.npy"), embedding)

    
    # Create output folder
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    # Load CNN model
    model = CNN(channels=channels, dropout=dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint = torch.load('cnn_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    # Make batches
    uniprots_batches, loader_set = making_loader(embeddings_dict, sequence_cut, batch_size)

# Predict
    predictions_proba = {}
    with torch.no_grad():
        for uniprots_batch, (inputs, mask) in zip(uniprots_batches, loader_set):
            outputs = model(inputs, mask)
            outputs = torch.sigmoid(outputs)
            outputs = outputs.detach().numpy()
            mask_np = mask.detach().numpy()
            for example in range(len(uniprots_batch)):
                valid_len = int(mask_np[example].sum())
                scores = outputs[example, :valid_len]
                # rispettare sequence_cut se necessario
                if valid_len > sequence_cut:
                    scores = scores[:sequence_cut]
                predictions_proba[uniprots_batch[example]] = scores

    # ...existing code...
    # Save predictions (single file)
    all_output_file = os.path.join(output_folder, 'all_predictions.caid')
    with open(all_output_file, 'w') as out_f:
        for protein_id in input_split.keys():
            if protein_id not in predictions_proba:
                print(f"Warning: no predictions for {protein_id} (missing embedding or skipped).")
                continue
            scores = predictions_proba[protein_id]
            seq = input_split[protein_id][:len(scores)]
            if len(seq) != len(scores):
                print(f"Warning: mismatch lengths for {protein_id}: seq={len(seq)} scores={len(scores)}")
            out_f.write(f'>{protein_id}\n')
            for i, (residue, score) in enumerate(zip(seq, scores)):
                out_f.write(f'{i+1}\t{residue}\t{score:.3f}\n')

if __name__ == "__main__":
    main()
