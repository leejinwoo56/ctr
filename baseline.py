import os
import random
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CFG = {
    "BATCH_SIZE": 4096,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "SEED": 42,
    "NUM_WORKERS": 10,
    "USE_AMP": True,
}

device = "cuda" if torch.cuda.is_available() else "cpu"
USE_CUDA = device == "cuda"

if USE_CUDA:
    torch.set_float32_matmul_precision("high")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class ClickDataset(Dataset):
    def __init__(self, df, feature_cols, seq_col, target_col=None, has_target=True):
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_col = seq_col
        self.target_col = target_col
        self.has_target = has_target

        self.X = self.df[self.feature_cols].astype(np.float32).fillna(0).values
        self.seq_strings = self.df[self.seq_col].astype(str).values

        self.seq_arrays = []
        for s in self.seq_strings:
            if s:
                arr = np.fromstring(s, sep=",", dtype=np.float32)
            else:
                arr = np.array([], dtype=np.float32)
            if arr.size == 0:
                arr = np.array([0.0], dtype=np.float32)
            self.seq_arrays.append(arr)

        if self.has_target:
            self.y = self.df[self.target_col].astype(np.float32).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        seq = torch.from_numpy(self.seq_arrays[idx])

        if self.has_target:
            y = torch.tensor(self.y[idx], dtype=torch.float32)
            return x, seq, y
        return x, seq


def collate_fn_train(batch):
    xs, seqs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths, ys


def collate_fn_infer(batch):
    xs, seqs = zip(*batch)
    xs = torch.stack(xs)
    seqs_padded = nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=0.0)
    seq_lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    seq_lengths = torch.clamp(seq_lengths, min=1)
    return xs, seqs_padded, seq_lengths


class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=64, hidden_units=(256, 128), dropout=0.2):
        super().__init__()
        self.bn_x = nn.BatchNorm1d(d_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True)

        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)

        x_seq = x_seq.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        h = h_n[-1]

        z = torch.cat([x, h], dim=1)
        return self.mlp(z).squeeze(1)


def make_loader(dataset, batch_size, shuffle, collate_fn):
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_fn,
        "num_workers": CFG["NUM_WORKERS"],
        "pin_memory": USE_CUDA,
    }
    if CFG["NUM_WORKERS"] > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 4
    return DataLoader(dataset, **kwargs)


def train_model(train_df, feature_cols, seq_col, target_col, batch_size=512, epochs=3, lr=1e-3):
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)

    model = TabularSeqModel(d_features=len(feature_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    amp_enabled = USE_CUDA and CFG["USE_AMP"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bi, (xs, seqs, seq_lens, ys) in enumerate(train_loader, 1):
            xs = xs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            seq_lens = seq_lens.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xs, seqs, seq_lens)
                loss = criterion(logits, ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * ys.size(0)
            print(f"[Train] epoch={epoch} batch={bi}/{len(train_loader)} loss={loss.item():.6f}", flush=True)

        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bi, (xs, seqs, seq_lens, ys) in enumerate(val_loader, 1):
                xs = xs.to(device, non_blocking=True)
                seqs = seqs.to(device, non_blocking=True)
                seq_lens = seq_lens.to(device, non_blocking=True)
                ys = ys.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=amp_enabled):
                    logits = model(xs, seqs, seq_lens)
                    loss = criterion(logits, ys)

                val_loss += loss.item() * len(ys)
                print(f"[Val]   epoch={epoch} batch={bi}/{len(val_loader)} loss={loss.item():.6f}", flush=True)

        val_loss /= len(val_dataset)

        if USE_CUDA:
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f} gpu_mem={mem_mb:.1f}MB", flush=True)
        else:
            print(f"[Epoch {epoch}] train={train_loss:.4f} val={val_loss:.4f}", flush=True)

    return model


def main():
    seed_everything(CFG["SEED"])

    all_train = pd.read_parquet("./train.parquet", engine="pyarrow")
    test = pd.read_parquet("./test.parquet", engine="pyarrow").drop(columns=["ID"])

    print("Train shape:", all_train.shape)
    print("Test shape:", test.shape)
    print("device:", device)
    if USE_CUDA:
        print("gpu:", torch.cuda.get_device_name(0))

    clicked_1 = all_train[all_train["clicked"] == 1]
    clicked_0 = all_train[all_train["clicked"] == 0].sample(n=len(clicked_1) * 2, random_state=42)
    train = pd.concat([clicked_1, clicked_0], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

    print("Train shape:", train.shape)
    print("Train clicked:0:", train[train["clicked"] == 0].shape)
    print("Train clicked:1:", train[train["clicked"] == 1].shape)

    target_col = "clicked"
    seq_col = "seq"
    feature_cols = [c for c in train.columns if c not in {target_col, seq_col, "ID"}]

    print("Num features:", len(feature_cols))
    print("Sequence:", seq_col)
    print("Target:", target_col)

    model = train_model(
        train_df=train,
        feature_cols=feature_cols,
        seq_col=seq_col,
        target_col=target_col,
        batch_size=CFG["BATCH_SIZE"],
        epochs=CFG["EPOCHS"],
        lr=CFG["LEARNING_RATE"],
    )

    test_ds = ClickDataset(test, feature_cols, seq_col, has_target=False)
    test_ld = make_loader(test_ds, batch_size=CFG["BATCH_SIZE"], shuffle=False, collate_fn=collate_fn_infer)

    amp_enabled = USE_CUDA and CFG["USE_AMP"]
    model.eval()
    outs = []
    with torch.no_grad():
        for bi, (xs, seqs, lens) in enumerate(test_ld, 1):
            xs = xs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xs, seqs, lens)
            outs.append(torch.sigmoid(logits).cpu())
            print(f"[Infer] batch={bi}/{len(test_ld)}", flush=True)

    test_preds = torch.cat(outs).numpy()

    submit = pd.read_csv("./sample_submission.csv")
    submit["clicked"] = test_preds
    submit.to_csv("./baseline_submit.csv", index=False)
    print("Saved: baseline_submit.csv")


if __name__ == "__main__":
    freeze_support()
    main()
