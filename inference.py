from typing import Optional, List, Dict, Any

import os

import torch
from torch.utils.data import DataLoader

from dataset import MMRDataset


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    mode: str = 'CPC',
) -> List[Dict[str, Any]]:
    if mode not in ('CT', 'CP', 'CPC'):
        raise ValueError(f"mode must be 'CT', 'CP', or 'CPC', got '{mode}'")

    model.eval()
    results = []

    for batch in data_loader:
        if mode == 'CT':
            images, labels, names = batch
            images = images.to(device)
            outputs = model(images)

        elif mode == 'CP':
            ct, pet, labels, names = batch
            ct = ct.to(device)
            pet = pet.to(device)
            outputs = model(ct, pet)

        elif mode == 'CPC':
            ct, pet, labels, names, clinic = batch
            ct = ct.to(device)
            pet = pet.to(device)
            clinic = clinic.to(device)
            outputs = model(ct, pet, clinic)

        probs = torch.sigmoid(outputs).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        for i in range(len(names)):
            prob_val = float(probs[i, 0])
            pred_val = int(preds[i, 0])
            label_val = int(labels[i].item()) if hasattr(labels[i], 'item') else int(labels[i])

            results.append({
                'id': names[i],
                'label': label_val,
                'probability': prob_val,
                'prediction': pred_val,
            })

    if len(results) == 0:
        print("Warning: no predictions generated, data_loader may be empty")

    return results


def load_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path}")

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'net' not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'net' key. Available keys: "
            f"{list(checkpoint.keys())}"
        )

    model.load_state_dict(checkpoint['net'])
    return model


if __name__ == '__main__':
    from models.fusion import CPC_ResNet18_mutli_attention
    from utils import set_seed

    set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = CPC_ResNet18_mutli_attention(in_channels=1, num_class=1)
    model = load_checkpoint(model, './result/best.ckpt', device=device)
    print('Model loaded successfully')
