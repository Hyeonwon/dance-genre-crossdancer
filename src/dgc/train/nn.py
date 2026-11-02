"""Transformer (+ 증강, + 장르 조건부 적대 학습) 학습/평가."""
import copy
import random
import time

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from dgc.config import window_len
from dgc.data.windows import frame_dim, genre_index, index_windows, window_input
from dgc.eval.metrics import compute_metrics, seq_aggregate
from dgc.eval.probe import dancer_probe
from dgc.models import build_backbone
from dgc.models.adversarial import GenreConditionalAdversary, dann_lambda


class WindowDataset(Dataset):
    def __init__(self, cache, names, cfg, g2i, dancer_idx, augment=False, seed=0):
        self.cache, self.cfg, self.g2i, self.dancer_idx = cache, cfg, g2i, dancer_idx
        self.items = index_windows(cache, names, cfg)
        self.augment = augment
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        name, start = self.items[i]
        rng = np.random.default_rng((self.seed, i, random.getrandbits(32))) if self.augment else None
        x = window_input(self.cache, name, start, self.cfg, rng, self.augment)
        s = self.cache["seqs"][name]
        return torch.from_numpy(x), self.g2i[s["genre"]], self.dancer_idx.get(s["dancer"], -1), i


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def within_genre_dancer_index(cache, train_names, g2i):
    """학습 댄서를 장르 안 번호(0..K-1)로 매긴다. 판별기는 이 번호를 맞히도록 학습된다."""
    per_genre = {g: sorted({cache["seqs"][n]["dancer"] for n in train_names
                            if cache["seqs"][n]["genre"] == g}) for g in g2i}
    k_max = max(len(v) for v in per_genre.values())
    idx, mask = {}, np.zeros((len(g2i), k_max), dtype=bool)
    for g, ds in per_genre.items():
        for k, d in enumerate(ds):
            idx[d] = k
            mask[g2i[g], k] = True
    return idx, mask, k_max


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y, D, I, Z = [], [], [], [], []
    for x, y, d, i in loader:
        logits, z = model(x.to(device))
        P.append(torch.softmax(logits, 1).cpu())
        Z.append(z.cpu())
        Y.append(y)
        D.append(d)
        I.append(i)
    cat = lambda t: torch.cat(t).numpy()
    return cat(P), cat(Y), cat(D), cat(I), cat(Z)


def _seq_macro_f1(P, Y, I, ds, g2i):
    seq = [ds.items[i][0] for i in I]
    names, sp = seq_aggregate(P, seq)
    y = np.array([g2i[ds.cache["seqs"][n]["genre"]] for n in names])
    return f1_score(y, sp.argmax(1), labels=list(range(len(g2i))), average="macro", zero_division=0)


def run_nn(cfg, cache, split, seed=0, log=print) -> dict:
    set_seed(seed)
    device = pick_device()
    n = cfg["nn"]
    g2i = genre_index(cache)
    d_idx, valid_mask, k_max = within_genre_dancer_index(cache, split["train"], g2i)

    ds_tr = WindowDataset(cache, split["train"], cfg, g2i, d_idx, augment=cfg["aug"]["enabled"], seed=seed)
    ds_tr_eval = WindowDataset(cache, split["train"], cfg, g2i, d_idx)
    ds_va = WindowDataset(cache, split["val"], cfg, g2i, d_idx)
    ds_te = WindowDataset(cache, split["test"], cfg, g2i, d_idx)
    kw = dict(batch_size=n["batch_size"], num_workers=n["num_workers"])
    dl_tr = DataLoader(ds_tr, shuffle=True, drop_last=True, **kw)
    dl_va, dl_te = DataLoader(ds_va, **kw), DataLoader(ds_te, **kw)

    model = build_backbone(cfg, frame_dim(cfg), len(g2i), window_len(cfg)).to(device)
    # 입력 표준화 통계 (증강 없는 학습 윈도우 일부로 계산)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(ds_tr_eval), min(n["stats_windows"], len(ds_tr_eval)), replace=False)
    xs = np.concatenate([ds_tr_eval[int(i)][0].numpy() for i in sample])
    model.set_input_stats(xs.mean(0), xs.std(0))

    adv = None
    if cfg["adv"]["enabled"]:
        adv = GenreConditionalAdversary(model.embed_dim, len(g2i), k_max, valid_mask,
                                        cfg["adv"]["hidden"]).to(device)
    groups = [{"params": model.parameters(), "lr": n["lr"]}]
    if adv:   # 판별기가 너무 약하면 인코더가 '쉽게 속이고' 끝나므로, 판별기는 더 빠르게 학습시킨다
        groups.append({"params": adv.parameters(), "lr": n["lr"] * cfg["adv"]["lr_mult"]})
    opt = torch.optim.AdamW(groups, weight_decay=n["weight_decay"])
    params_all = [p for g in opt.param_groups for p in g["params"]]
    ce = nn.CrossEntropyLoss()

    best, best_state, bad, history = -1.0, None, 0, []
    total = n["max_epochs"] * max(1, len(dl_tr))
    step = 0
    for epoch in range(n["max_epochs"]):
        t0 = time.time()
        model.train()
        if adv:
            adv.train()
        tot, tot_adv, nb = 0.0, 0.0, 0
        for x, y, d, _ in dl_tr:
            x, y, d = x.to(device), y.to(device), d.to(device)
            logits, z = model(x)
            loss = ce(logits, y)
            if adv:
                lam = cfg["adv"].get("lambda_max", 1.0) * dann_lambda(step / total, cfg["adv"]["gamma"])
                ok = d >= 0
                if ok.any():
                    adv_logits = adv(z[ok], y[ok], lam)
                    l_adv = ce(adv_logits, d[ok])
                    loss = loss + cfg["adv"]["weight"] * l_adv
                    tot_adv += l_adv.item()
            opt.zero_grad()
            loss.backward()
            if n.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params_all, n["grad_clip"])
            opt.step()
            tot += loss.item()
            nb += 1
            step += 1
        P, Y, _, I, _ = predict(model, dl_va, device)
        f1 = _seq_macro_f1(P, Y, I, ds_va, g2i)
        history.append({"epoch": epoch, "loss": tot / max(nb, 1),
                        "adv_loss": tot_adv / max(nb, 1) if adv else None, "val_seq_macro_f1": f1})
        log(f"  epoch {epoch:3d} loss {tot / max(nb, 1):.4f}"
            + (f" adv {tot_adv / max(nb, 1):.4f}" if adv else "")
            + f" val_f1 {f1:.4f} ({time.time() - t0:.1f}s)")
        if n.get("selection", "best_val") == "last":
            # 고정 epoch: 검증셋(학습과 같은 댄서)으로 모델을 고르지 않는다
            best = max(best, f1)
            continue
        if f1 > best:
            best, best_state, bad = f1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= n["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    P, Y, _, I, _ = predict(model, dl_te, device)
    test = compute_metrics(P, Y, [ds_te.items[i][0] for i in I], cache, g2i)

    # 개인 정보량: 학습 윈도우로 probe 학습 → 검증 윈도우(같은 댄서들의 다른 시퀀스)로 평가
    _, Ytr, Dtr, _, Ztr = predict(model, DataLoader(ds_tr_eval, **kw), device)
    _, Yva, Dva, _, Zva = predict(model, dl_va, device)
    probe = dancer_probe(Ztr, Ytr, Dtr, Zva, Yva, Dva, cfg["probe"]["max_windows_per_genre"], seed)

    return {"model": cfg.get("exp_name", "nn"), "seed": seed, "device": str(device),
            "best_val_seq_macro_f1": best, "epochs_run": len(history), "test": test,
            "probe": probe, "history": history,
            "n_train_windows": len(ds_tr), "n_test_windows": len(ds_te)}
