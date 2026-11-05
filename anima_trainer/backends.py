"""Anima uses only the official sd-scripts backend."""
from pathlib import Path
def backend(settings):
    return 'sd-scripts'
def selected_environment(env, settings):
    return {**env, 'backend':'sd-scripts'}
def source_dir(env):
    return env.get('sd_scripts','')
def environment_valid(env):
    source = Path(source_dir(env))
    return bool(env.get('python')) and Path(env['python']).is_absolute() and Path(env['python']).is_file() and source.is_absolute() and all((source / p).is_file() for p in ('anima_train_network.py','networks/lora_anima.py','configs/t5_old/tokenizer.json'))
