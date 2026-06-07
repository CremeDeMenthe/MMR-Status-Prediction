import os
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

from dataset import MMRDataset
from utils import set_seed, create_weighted_sampler, get_default_transform


def train_one_epoch(model, train_loader, optimizer, criterion, device, mode='CP'):
    if mode not in ('CT', 'CP', 'CPC'):
        raise ValueError(f"mode must be 'CT', 'CP', or 'CPC', got '{mode}'")

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    probabilities, targets = [], []
    tn = fp = tp = fn = 0

    for batch in train_loader:
        optimizer.zero_grad()

        if mode == 'CT':
            images, labels, _ = batch
            images = images.to(device)
        elif mode == 'CP':
            ct, pet, labels, _ = batch
            images = (ct.to(device), pet.to(device))
        elif mode == 'CPC':
            ct, pet, labels, _, clinic = batch
            images = (ct.to(device), pet.to(device), clinic.to(device))

        labels = labels.to(device).view(-1, 1).float()

        if mode == 'CT':
            outputs = model(images)
        elif mode == 'CP':
            outputs = model(*images)
        elif mode == 'CPC':
            outputs = model(*images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()

        probabilities.extend(probs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()

    if len(train_loader) == 0:
        raise ValueError("train_loader is empty")

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'loss': total_loss / len(train_loader),
        'accuracy': 100.0 * correct / total,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'auc': roc_auc,
    }


@torch.no_grad()
def evaluate(model, test_loader, criterion, device, mode='CP'):
    if mode not in ('CT', 'CP', 'CPC'):
        raise ValueError(f"mode must be 'CT', 'CP', or 'CPC', got '{mode}'")

    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    probabilities, targets = [], []
    tn = fp = tp = fn = 0

    for batch in test_loader:
        if mode == 'CT':
            images, labels, _ = batch
            images = images.to(device)
        elif mode == 'CP':
            ct, pet, labels, _ = batch
            images = (ct.to(device), pet.to(device))
        elif mode == 'CPC':
            ct, pet, labels, _, clinic = batch
            images = (ct.to(device), pet.to(device), clinic.to(device))

        labels = labels.to(device).view(-1, 1).float()

        if mode == 'CT':
            outputs = model(images)
        elif mode == 'CP':
            outputs = model(*images)
        elif mode == 'CPC':
            outputs = model(*images)

        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()

        probabilities.extend(probs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()

    if len(test_loader) == 0:
        raise ValueError("test_loader is empty")

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    return {
        'loss': total_loss / len(test_loader),
        'accuracy': 100.0 * correct / total,
        'specificity': specificity,
        'sensitivity': sensitivity,
        'auc': roc_auc,
    }


def train_single_model(
    model,
    train_csv,
    test_csv,
    mode='CPC',
    epochs=400,
    batch_size=16,
    lr=0.0001,
    save_dir='./result',
    experiment_name='experiment',
    save_auc_threshold=0.75,
    device=None,
):
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if lr <= 0:
        raise ValueError(f"lr must be positive, got {lr}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    set_seed(42)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()
    transform = get_default_transform()
    sampler = create_weighted_sampler(train_csv, 'MMR_label')

    train_set = MMRDataset(train_csv, mode=mode, transform=transform)
    test_set = MMRDataset(test_csv, mode=mode)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    log_path = os.path.join(save_dir, 'logs', f'{experiment_name}.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_df = pd.DataFrame(columns=[
        'epoch', 'train_loss', 'train_acc', 'train_spec', 'train_sen', 'train_auc',
        'test_loss', 'test_acc', 'test_spec', 'test_sen', 'test_auc',
    ])
    log_df.to_csv(log_path, index=False)

    model_path = os.path.join(save_dir, 'checkpoints', experiment_name)
    os.makedirs(model_path, exist_ok=True)

    best_auc = 0
    min_loss = float('inf')

    for epoch in range(epochs):
        start = time.perf_counter()

        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, mode)
        test_metrics = evaluate(model, test_loader, criterion, device, mode)

        elapsed = time.perf_counter() - start
        print(f'[Epoch {epoch+1}/{epochs}] ({elapsed:.1f}s) '
              f'Train loss={train_metrics["loss"]:.3f} acc={train_metrics["accuracy"]:.1f} '
              f'auc={train_metrics["auc"]:.3f} | '
              f'Test loss={test_metrics["loss"]:.3f} acc={test_metrics["accuracy"]:.1f} '
              f'auc={test_metrics["auc"]:.3f}')

        row = [epoch + 1,
               train_metrics['loss'], train_metrics['accuracy'],
               train_metrics['specificity'], train_metrics['sensitivity'], train_metrics['auc'],
               test_metrics['loss'], test_metrics['accuracy'],
               test_metrics['specificity'], test_metrics['sensitivity'], test_metrics['auc']]
        pd.DataFrame([row]).to_csv(log_path, mode='a', header=False, index=False)

        if test_metrics['auc'] >= save_auc_threshold and test_metrics['auc'] > best_auc:
            best_auc = test_metrics['auc']
            min_loss = test_metrics['loss']
            state = {'net': model.state_dict(), 'epoch': epoch + 1}
            ckpt_path = os.path.join(
                model_path,
                f'best_model_{epoch+1}_{test_metrics["auc"]:.2f}_{test_metrics["loss"]:.2f}.ckpt'
            )
            torch.save(state, ckpt_path)
            print(f'  -> Saved best AUC model: {ckpt_path}')

        if test_metrics['loss'] < min_loss:
            min_loss = test_metrics['loss']
            state = {'net': model.state_dict(), 'epoch': epoch + 1}
            ckpt_path = os.path.join(
                model_path,
                f'best_loss_model_{epoch+1}_{test_metrics["auc"]:.2f}_{test_metrics["loss"]:.2f}.ckpt'
            )
            torch.save(state, ckpt_path)
            print(f'  -> Saved best loss model: {ckpt_path}')


def train_fusion_model(
    model_P,
    model_C,
    model_fusion,
    train_csv,
    test_csv,
    external_csv=None,
    epochs=300,
    batch_size=16,
    lr=0.0001,
    save_dir='./result',
    experiment_name='fusion',
    device=None,
):
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test CSV not found: {test_csv}")
    if external_csv and not os.path.isfile(external_csv):
        raise FileNotFoundError(f"External CSV not found: {external_csv}")
    if epochs <= 0:
        raise ValueError(f"epochs must be positive, got {epochs}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_P.to(device)
    model_C.to(device)
    model_fusion.to(device)

    set_seed(42)

    all_params = list(model_P.parameters()) + list(model_C.parameters()) + list(model_fusion.parameters())
    optimizer = optim.SGD(all_params, lr=lr, momentum=0.9, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    transform = get_default_transform()
    sampler = create_weighted_sampler(train_csv, 'MMR_label')

    train_set = MMRDataset(train_csv, mode='CPC', transform=transform)
    test_set = MMRDataset(test_csv, mode='CPC')
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

    external_loader = None
    if external_csv:
        ext_set = MMRDataset(external_csv, mode='CPC')
        external_loader = DataLoader(ext_set, batch_size=batch_size, shuffle=True)

    log_path = os.path.join(save_dir, 'logs', f'{experiment_name}.csv')
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    columns = ['epoch', 'train_loss', 'train_acc', 'train_spec', 'train_sen', 'train_auc',
               'test_loss', 'test_acc', 'test_spec', 'test_sen', 'test_auc']
    if external_loader:
        columns += ['ex_loss', 'ex_acc', 'ex_spec', 'ex_sen', 'ex_auc']
    pd.DataFrame(columns=columns).to_csv(log_path, index=False)

    model_path = os.path.join(save_dir, 'checkpoints', experiment_name)
    os.makedirs(model_path, exist_ok=True)

    for epoch in range(epochs):
        start = time.perf_counter()

        model_P.train()
        model_C.train()
        model_fusion.train()
        total_loss = 0.0
        correct = total = 0
        probabilities, targets = [], []
        tn = fp = tp = fn = 0

        for batch in train_loader:
            ct, pet, labels, _, clinic = batch
            ct, pet, labels, clinic = ct.to(device), pet.to(device), labels.to(device), clinic.to(device)
            labels = labels.view(-1, 1).float()

            optimizer.zero_grad()
            feat_C = model_C(ct)
            feat_P = model_P(pet)
            outputs = model_fusion(feat_P, feat_C, clinic)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            probs = torch.sigmoid(outputs)
            predicted = (probs >= 0.5).float()
            probabilities.extend(probs.detach().cpu().numpy())
            targets.extend(labels.cpu().numpy())
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

        fpr, tpr, _ = roc_curve(targets, probabilities)
        train_auc = auc(fpr, tpr)
        train_metrics = {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'auc': train_auc,
        }

        test_metrics = _evaluate_fusion(model_P, model_C, model_fusion, test_loader, criterion, device)

        row = [epoch + 1,
               train_metrics['loss'], train_metrics['accuracy'],
               train_metrics['specificity'], train_metrics['sensitivity'], train_metrics['auc'],
               test_metrics['loss'], test_metrics['accuracy'],
               test_metrics['specificity'], test_metrics['sensitivity'], test_metrics['auc']]

        ext_metrics = None
        if external_loader:
            ext_metrics = _evaluate_fusion(model_P, model_C, model_fusion, external_loader, criterion, device)
            row += [ext_metrics['loss'], ext_metrics['accuracy'],
                    ext_metrics['specificity'], ext_metrics['sensitivity'], ext_metrics['auc']]

        elapsed = time.perf_counter() - start
        print(f'[Epoch {epoch+1}/{epochs}] ({elapsed:.1f}s) '
              f'Train auc={train_metrics["auc"]:.3f} | Test auc={test_metrics["auc"]:.3f}'
              + (f' | Ext auc={ext_metrics["auc"]:.3f}' if ext_metrics else ''))

        pd.DataFrame([row]).to_csv(log_path, mode='a', header=False, index=False)

        if ext_metrics and train_metrics['auc'] > test_metrics['auc'] and \
           train_metrics['auc'] > ext_metrics['auc'] and \
           ext_metrics['auc'] >= 0.72 and test_metrics['auc'] >= 0.84:
            state = {
                'net_P': model_P.state_dict(),
                'net_C': model_C.state_dict(),
                'model_fusion': model_fusion.state_dict(),
                'epoch': epoch + 1,
            }
            ckpt_path = os.path.join(
                model_path,
                f'best_model_{epoch+1}_{test_metrics["auc"]:.2f}_{test_metrics["loss"]:.2f}.ckpt'
            )
            torch.save(state, ckpt_path)
            print(f'  -> Saved model: {ckpt_path}')


@torch.no_grad()
def _evaluate_fusion(model_P, model_C, model_fusion, loader, criterion, device):
    if len(loader) == 0:
        raise ValueError("Evaluation loader is empty")

    model_P.eval()
    model_C.eval()
    model_fusion.eval()
    total_loss = 0.0
    correct = total = 0
    probabilities, targets = [], []
    tn = fp = tp = fn = 0

    for batch in loader:
        ct, pet, labels, _, clinic = batch
        ct, pet, labels, clinic = ct.to(device), pet.to(device), labels.to(device), clinic.to(device)
        labels = labels.view(-1, 1).float()

        feat_C = model_C(ct)
        feat_P = model_P(pet)
        outputs = model_fusion(feat_P, feat_C, clinic)
        loss = criterion(outputs, labels)

        probs = torch.sigmoid(outputs)
        predicted = (probs >= 0.5).float()
        probabilities.extend(probs.detach().cpu().numpy())
        targets.extend(labels.cpu().numpy())
        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        tn += ((predicted == 0) & (labels == 0)).sum().item()
        fp += ((predicted == 1) & (labels == 0)).sum().item()
        tp += ((predicted == 1) & (labels == 1)).sum().item()
        fn += ((predicted == 0) & (labels == 1)).sum().item()

    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    return {
        'loss': total_loss / len(loader),
        'accuracy': 100.0 * correct / total,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'auc': roc_auc,
    }


if __name__ == '__main__':
    from models.fusion import CPC_ResNet18_mutli_attention
    from utils import set_seed

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CPC_ResNet18_mutli_attention(in_channels=1, num_class=1)
    train_single_model(
        model=model,
        train_csv='./Dataset/train/train3_MMR.xlsx',
        test_csv='./Dataset/test/test3_MMR.xlsx',
        mode='CPC',
        epochs=400,
        batch_size=16,
        lr=0.0001,
        save_dir='./result',
        experiment_name='CPC_ResNet18_attention',
    )
