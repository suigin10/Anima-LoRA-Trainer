"""Project-owned cache paths, invalidated by source/model/settings changes."""
import hashlib
import json
from pathlib import Path


def identity(value):
    p = Path(value)
    if not value or not p.exists():
        return str(value)
    if p.is_dir():
        return [(str(f.resolve()), f.stat().st_size, f.stat().st_mtime_ns)
                for f in sorted(p.iterdir()) if f.is_file()]
    stat = p.stat()
    return [str(p.resolve()), stat.st_size, stat.st_mtime_ns]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def install_cache_paths(strategy, project, root):
    settings = project['settings']
    model, dataset = settings['model'], settings['dataset']
    signature = digest({'model':{k:identity(v) if isinstance(v,str) else v for k,v in model.items()},
                        'dataset': dataset, 'subsets':project['subsets'],
                        'vae_2d':settings['memory']['qwen_image_vae_2d'], 'schema':1})
    cache_root = Path(root) / signature
    extension = dataset['caption_extension']

    def location(image, original, text):
        image = Path(image)
        # Caption content participates, so editing text invalidates cached embeddings.
        caption = image.with_suffix(extension)
        if not caption.is_file():
            parts = str(image.with_suffix('')).split('_')
            if len(parts) >= 5:
                caption = Path('_'.join(parts[:-4]) + extension)
        caption_bytes = caption.read_bytes().hex() if text and caption.is_file() else ''
        key = digest([identity(image), caption_bytes])
        folder = cache_root / key
        folder.mkdir(parents=True, exist_ok=True)
        # Retain upstream's resolution-bearing filename for latent size parsing.
        return str(folder / Path(original).name)

    old_te = strategy.AnimaTextEncoderOutputsCachingStrategy.get_outputs_npz_path
    old_latent = strategy.AnimaLatentsCachingStrategy.get_latents_npz_path
    def text_path(self, image):
        return location(image, old_te(self,image), True)
    def latent_path(self, image, size):
        return location(image, old_latent(self,image,size), False)
    strategy.AnimaTextEncoderOutputsCachingStrategy.get_outputs_npz_path = text_path
    strategy.AnimaLatentsCachingStrategy.get_latents_npz_path = latent_path
