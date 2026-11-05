"""Read safetensors headers only; GUI never loads tensors or pickle files."""
import json
from pathlib import Path
import re
import struct
from .i18n import tr


def header(path):
    with Path(path).open('rb') as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError('Invalid safetensors header')
        count = struct.unpack('<Q', size)[0]
        if not 2 <= count <= 64 * 1024 * 1024:
            raise ValueError('Invalid safetensors header size')
        data = stream.read(count)
        if len(data) != count:
            raise ValueError('Truncated safetensors header')
        value = json.loads(data)
        if not isinstance(value, dict):
            raise ValueError('Invalid safetensors header')
        return {normalize(k): v for k, v in value.items() if k != '__metadata__'}


def normalize(key):
    for prefix in ('model.diffusion_model.', 'net.'):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def model_errors(settings):
    model = settings['model']
    path = Path(model['pretrained_model_name_or_path'])
    if not path.is_file():
        return []  # Missing paths are reported by the ordinary path checks.
    errors = []
    try:
        weights = header(path)
        blocks = {int(m[1]) for key in weights if (m := re.match(r'blocks\.(\d+)\.', key))}
        if blocks == set(range(40)):
            errors.append(tr('Anima 2.9B（40層）のLoRA学習は非対応です。28層のAnimaを指定してください。'))
        elif blocks != set(range(28)):
            errors.append(tr('このsd-scripts版は28ブロックのAnimaに対応します。モデルを確認してください。'))
        if not any('self_attn' in key for key in weights) or not any('cross_attn' in key for key in weights):
            errors.append(tr('Anima DiTのAttention重みを確認できません。'))
        if any(str(v.get('dtype', '')).startswith(('F8','I','U')) for k,v in weights.items() if k.endswith('.weight') and isinstance(v, dict)):
            errors.append(tr('量子化済みDiTは非対応です。bf16 / fp16のAnimaを指定してください。'))
        if settings['memory']['blocks_to_swap'] > max(0, len(blocks)-2):
            errors.append(tr('Block swapはこのモデルでは0〜{p0}にしてください。', p0=max(0,len(blocks)-2)))
        adapter = model['llm_adapter_path']
        if adapter:
            if Path(adapter).is_file() and 'llm_adapter.out_proj.weight' not in header(adapter):
                errors.append(tr('LLM Adapterファイルにllm_adapter.out_proj.weightがありません。'))
        elif 'llm_adapter.out_proj.weight' not in weights:
            errors.append(tr('DiTにLLM Adapterがありません。別ファイルを指定してください。'))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(tr('Animaモデルのヘッダーを読めません: {p0}', p0=exc))
    return errors
