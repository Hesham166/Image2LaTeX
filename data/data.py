import os
import csv
import glob
import shutil
import logging
from typing import Optional, Tuple
from dataclasses import dataclass

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import kagglehub


logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    """Configuration for dataset loading and processing."""
    batch_size: int = 32
    img_size: Tuple[int, int] = (128, 512)
    dataset_handle: str = "shahrukhkhan/im2latex100k"
    local_dir: str = "../data/im2latex-100k"
    pad_fill: int = 255
    force_download: bool = False


def ensure_dataset(config: DataConfig) -> str:
    """Download and cache dataset locally."""
    logger.info(f"Ensuring dataset {config.dataset_handle} is available...")

    cache_path = kagglehub.dataset_download(config.dataset_handle, force_download=config.force_download)
    target = os.path.abspath(config.local_dir)

    if not os.path.exists(target) or not os.lsitdir(target):
        os.makedirs(target, exist_ok=True)
        logger.info(f"Copying dataset to {target}")

        if os.path.isdir(cache_path):
            for item in os.listdir(cache_path):
                src, dst = os.path.join(cache_path, item), os.path.join(target, item)
                (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        else:
            shutil.copy2(cache_path, target)
    
    return target


class Vocab:
    """Vocabulary for token-to-index mapping."""
    
    def __init__(self, texts, specials=None):
        specials = specials or ['<pad>', '<sos>', '<eos>', '<unk>']
        tokens = {token for text in texts for token in text.strip().split()}
        
        self.itos = specials + sorted(tokens)
        self.stoi = {token: i for i, token in enumerate(self.itos)}
        
        # Cache special token indices
        for special in specials:
            setattr(self, f"{special[1:-1]}_idx", self.stoi[special])
    
    def encode(self, text: str) -> list[int]:
        tokens = text.strip().split()
        ids = [self.stoi.get(token, self.unk_idx) for token in tokens]
        return [self.sos_idx] + ids + [self.eos_idx]
    
    def decode(self, ids: list[int]) -> str:
        tokens = [self.itos[i] for i in ids if i not in {self.sos_idx, self.eos_idx, self.pad_idx}]
        return " ".join(tokens)
    
    def __len__(self) -> int:
        return len(self.itos)


class ResizeWithPad:
    """Resize image preserving aspect ratio, then pad to target size."""

    def __init__(self, target_size: Tuple[int, int], fill: int = 255):
        self.target_h, self.target_w = target_size
        self.fill = fill

    def __call__(self, img: Image.Image) -> torch.Tensor:
        w, h = img.size
        scale = min(self.target_w / w, self.target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        img = TF.resize(img, (new_h, new_w))
        
        # Center padding
        pad_w, pad_h = self.target_w - new_w, self.target_h - new_h
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        
        return TF.to_tensor(TF.pad(img, padding, fill=self.fill))


class Im2LatexDataset(Dataset):
    """Dataset for image-to-LaTeX conversion."""
    
    def __init__(self, split: str, path: str, config: DataConfig, vocab: Optional[Vocab] = None, skip_missing: bool = True):
        self.config = config
        self.skip_missing = skip_missing
        
        split_file = os.path.join(path, f'im2latex_{split}.csv')
        try:
            with open(split_file, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                self.samples = [(row[0].strip(), row[1].strip()) for row in reader if len(row) == 2]
        except (FileNotFoundError, IOError) as e:
            raise FileNotFoundError(f"Cannot read {split_file}: {e}")
        
        if vocab is None:
            texts = [formula for formula, _ in self.samples]
            self.vocab = Vocab(texts)
        else:
            self.vocab = vocab
        
        self.transform = ResizeWithPad(config.img_size, config.pad_fill)
        
        img_dirs = glob.glob(os.path.join(path, "formula_images_processed/formula_images_processed*"))
        if not img_dirs:
            raise FileNotFoundError(f"No formula_images_processed directory found in {path}")
        self.img_root = img_dirs[0]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        formula, image_name = self.samples[idx]
        img_path = os.path.join(self.img_root, image_name)
        
        try:
            img = Image.open(img_path).convert("L")
            img_tensor = self.transform(img)
            tgt_tensor = torch.tensor(self.vocab.encode(formula), dtype=torch.long)
            return img_tensor, tgt_tensor
        except (FileNotFoundError, IOError, OSError) as e:
            if self.skip_missing:
                logger.warning(f"Skipping {img_path}: {e}")
                return None
            raise RuntimeError(f"Failed to load {img_path}: {e}")


def collate_fn(batch, pad_idx: int = 0):
    """Collate function that filters None entries and pads sequences."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    
    imgs, tgts = zip(*batch)
    imgs = torch.stack(imgs)
    tgts = torch.nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_idx)
    return imgs, tgts


def get_loaders(config: DataConfig = None):
    """Create train/validation/test DataLoaders with shared vocabulary."""
    config = config or DataConfig()
    
    if os.path.exists(config.local_dir):
        path = config.local_dir
    else:
        path = ensure_dataset(config)
    
    logger.info(f"Dataset ready at: {path}")
    
    train_dataset = Im2LatexDataset("train", path, config)
    vocab = train_dataset.vocab
    
    val_dataset = Im2LatexDataset("validate", path, config, vocab)
    test_dataset = Im2LatexDataset("test", path, config, vocab)
    
    loader_kwargs = {
        'batch_size': config.batch_size,
        'collate_fn': lambda b: collate_fn(b, vocab.pad_idx)
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)
    
    return train_loader, val_loader, test_loader, vocab


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    logging.basicConfig(level=logging.INFO)
    
    config = DataConfig(batch_size=16)
    train_loader, val_loader, test_loader, vocab = get_loaders(config)
    
    print(f"Datasets: {len(train_loader.dataset)}, {len(val_loader.dataset)}, "
          f"{len(test_loader.dataset)}, vocab: {len(vocab)}")
    
    plt.figure(figsize=(20, 8))
    for idx, (imgs, tgts) in enumerate(train_loader):
        if idx >= 6: break
        plt.subplot(2, 3, idx + 1)
        plt.imshow(imgs[0, 0], cmap='gray')
        plt.title(vocab.decode(tgts[0][:20].tolist()), fontsize=8)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()