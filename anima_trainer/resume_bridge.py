"""Compatibility fixes for sd-scripts' cumulative resume counters.

Only source text is inspected here; checkpoint deserialization remains Accelerate's job.
The installed sd-scripts checkout is never modified.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import runpy
import sys
import tomllib


def patch_source(source, recalculate_scheduler=False):
    tree = ast.parse(source)
    trainer = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'NetworkTrainer'), None)
    train = next((n for n in trainer.body if isinstance(n, ast.FunctionDef) and n.name == 'train'), None) if trainer else None
    if train is None:
        raise RuntimeError('未対応の sd-scripts: NetworkTrainer.train が見つかりません')
    lines = source.splitlines(keepends=True)
    edits = []
    def replace(node, text):
        edits.append((node.lineno - 1, node.end_lineno, text))
    def assign(node, name):
        return isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    if recalculate_scheduler:
        resume = [n for n in train.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                  and ast.unparse(n.value.func) == 'args_util.resume_from_local_or_hf_if_specified']
        if len(resume) != 1:
            raise RuntimeError('未対応の sd-scripts: scheduler復元処理が変更されています')
        node = resume[0]
        edits.append((node.end_lineno, node.end_lineno,
                      '        if args.resume:\n'
                      '            _anima_realign_scheduler(lr_scheduler, steps_from_state)\n'))
    start = next((i for i, n in enumerate(train.body) if assign(n, 'epoch_to_start')), None)
    if start is None or start + 2 >= len(train.body) or not assign(train.body[start + 2], 'global_step'):
        raise RuntimeError('未対応の sd-scripts: 再開位置の処理が変更されています')
    old = train.body[start + 1]
    if not isinstance(old, ast.If) or 'skip_until_initial_step' not in ast.unparse(old):
        raise RuntimeError('未対応の sd-scripts: バッチスキップ処理が変更されています')
    edits.append((train.body[start].lineno - 1, train.body[start + 2].end_lineno,
        '        resume_global_step = initial_step\n'
        '        epoch_to_start = resume_global_step // num_update_steps_per_epoch\n'
        '        initial_step = ((resume_global_step % num_update_steps_per_epoch) * args.gradient_accumulation_steps\n'
        '                        if args.skip_until_initial_step else 0)\n'
        '        global_step = resume_global_step\n'
        '        current_step.value = global_step\n'))
    skip = [n for n in train.body if isinstance(n, ast.If) and
            any(assign(x, 'global_step') for x in n.body) and 'skip_epoch' in ast.unparse(n)]
    if len(skip) != 1:
        raise RuntimeError('未対応の sd-scripts: 累積 STEP の処理が変更されています')
    replace(skip[0], '')
    progress = [n for n in train.body if assign(n, 'progress_bar')]
    if len(progress) != 1 or 'args.max_train_steps - initial_step' not in ast.unparse(progress[0]):
        raise RuntimeError('未対応の sd-scripts: 進捗表示の処理が変更されています')
    replace(progress[0], '        progress_bar = tqdm(total=args.max_train_steps, initial=resume_global_step, smoothing=0, disable=not accelerator.is_local_main_process, desc="steps")\n')
    hook = next((n for n in train.body if isinstance(n, ast.FunctionDef) and n.name == 'save_model_hook'), None)
    dumps = [n for n in ast.walk(hook) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
             and ast.unparse(n.value.func) == 'json.dump'] if hook else []
    if len(dumps) != 1 or 'current_step.value + 1' not in ast.unparse(dumps[0]):
        raise RuntimeError('未対応の sd-scripts: state 保存処理が変更されています')
    replace(dumps[0], '                json.dump({"current_epoch": current_epoch.value, "current_step": global_step}, f)\n')
    logs = [n for n in hook.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and ast.unparse(n.value.func) == 'logger.info' and 'current_step.value + 1' in ast.unparse(n)]
    for log in logs:
        replace(log, '            logger.info(f"save train state to {train_state_file} at epoch {current_epoch.value} step {global_step}")\n')
    for begin, end, replacement in sorted(edits, reverse=True):
        lines[begin:end] = [replacement]
    result = ''.join(lines)
    compile(result, '<anima-resume-bridge>', 'exec')
    return result


def main():
    backend = Path(sys.argv.pop(1)).resolve()
    keep_latest = False
    recalculate_scheduler = False
    if sys.argv[1:2] == ['--state-policy']:
        policy = json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
        del sys.argv[1:3]
        keep_latest = policy.get('keep_latest_state_only', False)
        recalculate_scheduler = policy.get('recalculate_scheduler', False)
        if type(keep_latest) is not bool or type(recalculate_scheduler) is not bool:
            raise ValueError('Invalid state retention policy')
    sys.path.insert(0, str(backend))
    # Isolate caches from teacher images and from other model configurations.
    config_index = sys.argv.index('--config_file')
    config_path = Path(sys.argv[config_index + 1])
    config = tomllib.loads(config_path.read_text(encoding='utf-8'))
    snapshot = config_path.parent / 'project_snapshot.json'
    if snapshot.is_file():
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from anima_trainer.cache_paths import install_cache_paths
        from library import strategy_anima
        project = json.loads(snapshot.read_text(encoding='utf-8'))
        install_cache_paths(strategy_anima, project, config_path.parent.parent.parent / 'cache')
    # Upstream parses llm_adapter_path but its pinned loader does not apply it.
    # Add that optional overlay in memory; leave the installed checkout untouched.
    if config.get('llm_adapter_path'):
        adapter_path = config['llm_adapter_path']
        adapter_source = (backend / 'library/anima_utils.py').read_text(encoding='utf-8-sig')
        needle = '    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)'
        if adapter_source.count(needle) != 1:
            raise RuntimeError('Unsupported Anima loader: LLM Adapter overlay hook changed')
        adapter_source = adapter_source.replace(needle,
            '    _external = load_file(' + repr(adapter_path) + ', device="cpu")\n'
            '    for _key, _value in _external.items():\n'
            '        _key = rename_hook(_key)\n'
            '        if not _key.startswith("llm_adapter."):\n'
            '            raise ValueError("External adapter contains non-adapter weights")\n'
            '        sd[_key] = _value.to(device=loading_device, dtype=dit_weight_dtype)\n' + needle)
        adapter_spec = importlib.util.spec_from_file_location('library.anima_utils', backend / 'library/anima_utils.py')
        adapter_module = importlib.util.module_from_spec(adapter_spec)
        sys.modules['library.anima_utils'] = adapter_module
        exec(compile(adapter_source, str(backend / 'library/anima_utils.py'), 'exec'), adapter_module.__dict__)
        import library
        library.anima_utils = adapter_module
    path = backend / 'train_network.py'
    source = patch_source(path.read_text(encoding='utf-8-sig'), recalculate_scheduler)
    spec = importlib.util.spec_from_file_location('train_network', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['train_network'] = module
    exec(compile(source, str(path), 'exec'), module.__dict__)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    if recalculate_scheduler:
        from anima_trainer.resume_schedule import realign_scheduler
        module._anima_realign_scheduler = realign_scheduler
    if keep_latest:
        # This script is launched by absolute path in the backend environment.
        # Add only the GUI package root to import the stdlib-only retention helper.
        from anima_trainer.state_retention import install_state_retention
        from library import checkpoint_io
        install_state_retention(checkpoint_io)
        print('Anima LoRA Trainer: keep latest completed state only; LoRA saves are unchanged', flush=True)
    print('Anima LoRA Trainer: cumulative STEP / resume compatibility enabled', flush=True)
    runpy.run_path(str(backend / 'anima_train_network.py'), run_name='__main__')


if __name__ == '__main__':
    main()

