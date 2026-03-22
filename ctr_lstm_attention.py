import os
import random
from multiprocessing import freeze_support

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


CFG = {
    "BATCH_SIZE": 4096,
    "EPOCHS": 20,
    "LEARNING_RATE": 1e-3,
    "WEIGHT_DECAY": 1e-4,
    "SEED": 42,
    "NUM_WORKERS": 10,
    "USE_AMP": True,
    "EARLY_STOPPING_PATIENCE": 3,
    "GRAD_CLIP": 1.0,
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


class SeqAttentionPool(nn.Module):
    """LSTM 전체 hidden states에 대해 attention pooling."""
    def __init__(self, hidden_size):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states, seq_lengths):
        # hidden_states: (B, T, H)
        scores = self.attn(hidden_states).squeeze(-1)  # (B, T)

        # padding 위치 마스킹
        max_len = hidden_states.size(1)
        mask = torch.arange(max_len, device=hidden_states.device).unsqueeze(0) >= seq_lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        pooled = (hidden_states * weights).sum(dim=1)          # (B, H)
        return pooled


class TabularSeqModel(nn.Module):
    def __init__(self, d_features, lstm_hidden=64, hidden_units=(256, 128), dropout=0.3):
        super().__init__()
        self.bn_x = nn.BatchNorm1d(d_features)
        self.lstm = nn.LSTM(input_size=1, hidden_size=lstm_hidden, batch_first=True, num_layers=2, dropout=dropout)
        self.attn_pool = SeqAttentionPool(lstm_hidden)

        input_dim = d_features + lstm_hidden
        layers = []
        for h in hidden_units:
            layers += [nn.Linear(input_dim, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            input_dim = h
        layers += [nn.Linear(input_dim, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_feats, x_seq, seq_lengths):
        x = self.bn_x(x_feats)

        x_seq = x_seq.unsqueeze(-1)
        packed = nn.utils.rnn.pack_padded_sequence(
            x_seq, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)  # (B, T, H)
        h = self.attn_pool(hidden_states, seq_lengths)

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


def train_one_epoch(model, loader, criterion, optimizer, scaler, amp_enabled):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  Train", leave=False)
    for xs, seqs, seq_lens, ys in pbar:
        xs = xs.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True)
        seq_lens = seq_lens.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), CFG["GRAD_CLIP"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * ys.size(0)
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, amp_enabled):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []
    for xs, seqs, seq_lens, ys in tqdm(loader, desc="  Val  ", leave=False):
        xs = xs.to(device, non_blocking=True)
        seqs = seqs.to(device, non_blocking=True)
        seq_lens = seq_lens.to(device, non_blocking=True)
        ys = ys.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(xs, seqs, seq_lens)
            loss = criterion(logits, ys)

        total_loss += loss.item() * ys.size(0)
        all_probs.append(torch.sigmoid(logits).cpu())
        all_labels.append(ys.cpu())

    avg_loss = total_loss / len(loader.dataset)
    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    auc = roc_auc_score(labels, probs)
    return avg_loss, auc


def train_model(train_df, feature_cols, seq_col, target_col, batch_size=512, epochs=3, lr=1e-3):
    tr_df, va_df = train_test_split(train_df, test_size=0.2, random_state=42, shuffle=True)

    train_dataset = ClickDataset(tr_df, feature_cols, seq_col, target_col, has_target=True)
    val_dataset = ClickDataset(va_df, feature_cols, seq_col, target_col, has_target=True)

    train_loader = make_loader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_train)
    val_loader = make_loader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)

    model = TabularSeqModel(d_features=len(feature_cols)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=CFG["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    amp_enabled = USE_CUDA and CFG["USE_AMP"]
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    best_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler, amp_enabled)
        val_loss, val_auc = evaluate(model, val_loader, criterion, amp_enabled)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        log = f"[Epoch {epoch:02d}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_auc={val_auc:.4f}  lr={lr_now:.2e}"
        if USE_CUDA:
            mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            log += f"  gpu={mem_mb:.0f}MB"
        print(log, flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            print(f"  => Best AUC updated: {best_auc:.4f}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= CFG["EARLY_STOPPING_PATIENCE"]:
                print(f"Early stopping at epoch {epoch} (patience={CFG['EARLY_STOPPING_PATIENCE']})", flush=True)
                break

    model.load_state_dict(best_state)
    print(f"Loaded best model (val_auc={best_auc:.4f})", flush=True)
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

    print(f"Resampled train: total={len(train)}  click=1:{(train['clicked']==1).sum()}  click=0:{(train['clicked']==0).sum()}")

    target_col = "clicked"
    seq_col = "seq"
    feature_cols = [c for c in train.columns if c not in {target_col, seq_col, "ID"}]

    print(f"Num features: {len(feature_cols)},  seq_col: {seq_col},  target: {target_col}")

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
        for xs, seqs, lens in tqdm(test_ld, desc="Inference"):
            xs = xs.to(device, non_blocking=True)
            seqs = seqs.to(device, non_blocking=True)
            lens = lens.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(xs, seqs, lens)
            outs.append(torch.sigmoid(logits).cpu())

    test_preds = torch.cat(outs).numpy()

    submit = pd.read_csv("./sample_submission.csv")
    submit["clicked"] = test_preds
    submit.to_csv("./baseline_submit.csv", index=False)
    print("Saved: baseline_submit.csv")


if __name__ == "__main__":
    freeze_support()
    main()
