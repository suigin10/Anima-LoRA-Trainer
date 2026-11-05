from __future__ import annotations

from .i18n import tr
from .backends import backend, selected_environment, environment_valid, source_dir

import copy
from .anima_options import validate_anima, component_network_args
from .optimization import validate_optimization, apply_optimization, DEPENDENCIES, is_schedulefree
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
from datetime import datetime
from uuid import uuid4

SCHEMA_VERSION = 1
PROJECT_KIND = 'AnimaLoRATrainer'
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}

# UI and serialization share this schema. CLI-only constants are in training_config.
# key: (Japanese label, default, optional choices, hint)
FIELDS = {'model': {'pretrained_model_name_or_path': ('Anima DiT',
                                             '',
                                             'file',
                                             '28層のAnima用。Anima 2.9B（40層）は非対応。非量子化safetensorsを指定'),
           'qwen3': ('Qwen3-0.6B', '', 'model', '単体safetensorsまたは完全なHugging Faceモデルフォルダの、どちらか一方を指定'),
           'vae': ('Qwen-Image VAE', '', 'file', 'Anima用のQwen-Image VAEを指定'),
           'qwen3_max_token_length': ('Qwen3 最大トークン数', 512, None, 'キャプションを変更したときはキャッシュを再作成します'),
           't5_max_token_length': ('T5 最大トークン数', 512, None, ''),
           'llm_adapter_path': ('LLM Adapter（任意）', '', 'file', '空欄ならDiT内から取得。内蔵されていない場合のみ指定'),
           't5_tokenizer_path': ('T5 tokenizer（任意）', '', 'dir', '空欄ならsd-scripts同梱版。T5モデル本体は不要')},
 'dataset': {'resolution': ('学習解像度', 1024, None, '16の倍数。画像の解像度・GPUメモリに合わせて調整'),
             'batch_size': ('バッチサイズ', 1, None, 'VRAM 12GB のビデオカード向け開始値は 1'),
             'enable_bucket': ('アスペクト比 bucket', True, None, '縦横比ごとに学習解像度を調整'),
             'bucket_reso_steps': ('Bucket 刻み', 64, None, '16 の倍数'),
             'min_bucket_reso': ('Bucket 最小解像度', 512, None, ''),
             'max_bucket_reso': ('Bucket 最大解像度', 1536, None, ''),
             'bucket_no_upscale': ('小さい画像を拡大しない', True, None, ''),
             'caption_extension': ('キャプション拡張子', '.txt', None, '初期値は .txt。実際のキャプション拡張子に合わせます（例 .caption）。'),
             'enable_multiline_caption': ('複数行キャプション', True, None, '自然文キャプションの改行を保持'),
             'caption_dropout_rate': ('Caption dropout', 0.05, None, 'キャプション全体のdropout。TEキャッシュと併用可能'),
             'shuffle_caption': ('キャプションをシャッフル', False, None, 'TE キャッシュと併用不可。自然文では通常 OFF'),
             'color_aug': ('色の拡張', False, None, 'Latent キャッシュと併用不可'),
             'random_crop': ('ランダムクロップ', False, None, 'Latent キャッシュと併用不可'),
             'prior_loss_weight': ('正則化画像のLoss重み', 1.0, None, '正則化サブセットだけに適用。教師画像には適用しません')},
 'training': {'output_name': ('出力 LoRA 名', 'anima_lora', None, '拡張子不要'),
              'network_dim': ('Rank / dim', 16, None, 'state再開時は変更できません'),
              'network_alpha': ('Alpha', 16.0, None, 'RankとAlphaの比率でLoRAの倍率が変わります。state再開時は変更不可'),
              'learning_rate': ('学習率', 0.00005, None, '基本LR。層別LRが空欄の系統に適用'),
              'optimizer_type': ('Optimizer',
                                 'AdamW8bit',
                                 ['AdamW',
                                  'AdamW8bit',
                                  'PagedAdamW8bit',
                                  'PagedAdamW32bit',
                                  'Lion',
                                  'Lion8bit',
                                  'PagedLion8bit',
                                  'SGDNesterov',
                                  'CAME',
                                  'Adafactor',
                                  'Prodigy',
                                  'DAdaptAdam',
                                  'AdamWScheduleFree',
                                  'RAdamScheduleFree'],
                                 ''),
              'weight_decay': ('Weight decay', '0.01', None, '空欄ならOptimizerの既定値'),
              'lr_scheduler': ('LR scheduler',
                               'cosine',
                               ['constant',
                                'constant_with_warmup',
                                'linear',
                                'cosine',
                                'cosine_with_restarts',
                                'cosine_with_min_lr',
                                'polynomial',
                                'inverse_sqrt',
                                'warmup_stable_decay',
                                'adafactor'],
                               '選択に応じた詳細だけを有効化します。入力済みの学習率は変更しません'),
              'lr_scheduler_num_cycles': ('Restart周期数', 1, None, 'cosine_with_restarts専用'),
              'lr_scheduler_power': ('Polynomial power', 1.0, None, 'polynomial専用。最終学習率は追加引数lr_endで指定可能'),
              'lr_warmup_steps': ('Warmup steps', 0, None, 'constantでは使用しません'),
              'max_train_epochs': ('学習エポック数', 15, None, '最大stepsが0の場合に使用'),
              'max_train_steps': ('最大 steps（0: エポック指定）', 2000, None, '総STEP数。batch・勾配累積が増えると、1 STEPで見る画像数も増えます'),
              'gradient_accumulation_steps': ('勾配累積', 1, None, '実効バッチ数 = バッチサイズ × 勾配累積'),
              'max_grad_norm': ('勾配クリップ', 1.0, None, 'Adafactor 選択時は書き出し時に 0 を使用'),
              'save_every_n_epochs': ('何エポックごとに保存（0: 無効）', 0, None, ''),
              'save_every_n_steps': ('何 steps ごとに保存（0: 無効）', 100, None, ''),
              'save_state': ('再開用の optimizer state も保存', True, None, 'LoRA単体保存とは別です。stateは容量を多く使用します'),
              'keep_latest_state_only': ('state は最新の1つだけ保持',
                                         True,
                                         None,
                                         '新しいstateの保存成功後に、この実行で作成した古いstateだけを削除。LoRAは削除しません'),
              'save_state_on_train_end': ('正常終了時にも state を保存',
                                          True,
                                          None,
                                          '強制停止時には保存できません。直近の保存済み state から再開'),
              'skip_until_initial_step': ('再開時に処理済みバッチをスキップ', True, None, 'state の進捗とデータ位置を引き継ぐため通常 ON'),
              'seed': ('乱数 seed', 42, None, ''),
              'network_weights': ('初期 LoRA 重み（任意）', '', 'file', 'LoRAを初期値として新しいOptimizerで学習。state再開とは併用不可'),
              'resume': ('再開 state フォルダ（任意）', '', 'dir', 'コピー作成時はリセット。通常の新規学習では空欄'),
              'optimizer_args': ('Optimizer 追加引数',
                                 '',
                                 'multiline',
                                 '1行に1つ。例: betas=(0.9, 0.99) / eps=1e-8。LR・weight decayは専用欄で指定'),
              'lr_scheduler_args': ('Scheduler 追加引数',
                                    '',
                                    'multiline',
                                    '1行に1つ。例: lr_end=1e-7（polynomial）。専用欄のある引数はここへ重複指定しません'),
              'lr_scheduler_min_lr_ratio': ('最小学習率の比率',
                                            0.0,
                                            None,
                                            'cosine_with_min_lr / warmup_stable_decay。0〜1'),
              'lr_scheduler_timescale': ('Inverse sqrt timescale（0: 自動）',
                                         0,
                                         None,
                                         'inverse_sqrt専用。0ならWarmup STEP数を使用'),
              'lr_decay_steps': ('減衰STEP数', 0, None, 'warmup_stable_decay専用。Warmup後の一定期間を経て、このSTEP数で減衰'),
              'network_dropout': ('Neuron dropout', 0.0, None, 'LoRAの中間出力のdropout。0で無効'),
              'scale_weight_norms': ('LoRA weight norm上限（0: 無効）', 0.0, None, '上限を超えたLoRA重みを学習中に縮小する正則化')},
 'memory': {'blocks_to_swap': ('CPU にスワップするブロック数', 0, None, '0で無効。最大値はモデルから検出。CPU / Unsloth offloadと排他'),
            'gradient_checkpointing': ('Gradient checkpointing', True, None, 'メモリ削減のため通常 ON'),
            'cpu_offload_checkpointing': ('CPU offload checkpointing',
                                          False,
                                          None,
                                          'Block swap / Unsloth と併用不可'),
            'unsloth_offload_checkpointing': ('Unsloth offload checkpointing',
                                              False,
                                              None,
                                              'Block swap / CPU offload と併用不可'),
            'cache_latents': ('Latent キャッシュ', True, None, ''),
            'cache_latents_to_disk': ('Latent をディスクに保存', True, None, ''),
            'cache_text_encoder_outputs': ('テキスト出力キャッシュ', True, None, 'Qwen は凍結し、キャッシュ後 GPU から解放'),
            'cache_text_encoder_outputs_to_disk': ('テキスト出力をディスクに保存', True, None, ''),
            'attn_mode': ('Attention',
                          'torch',
                          ['torch', 'xformers', 'flash'],
                          '通常は torch を推奨。xformers / flash は追加パッケージが必要'),
            'split_attn': ('Split attention', False, None, 'xformers 選択時は自動で ON になり、必須のため固定されます'),
            'max_data_loader_n_workers': ('DataLoader workers', 4, None, 'CPU・RAMに合わせて調整。0ならメインプロセスで読み込み'),
            'num_cpu_threads_per_process': ('CPU処理スレッド数',
                                            2,
                                            None,
                                            'DataLoader workersとは別設定で、増やしても速くなるとは限りません'),
            'persistent_data_loader_workers': ('Worker をエポック間で維持', True, None, 'workers=0 の場合は OFF'),
            'compile': ('torch.compile', False, None, '追加環境が必要。まず OFF で動作を確認'),
            'qwen_image_vae_2d': ('画像専用2D VAE', True, None, '同じ公式VAE重みを読み込み時に変換。画像のキャッシュ準備を軽量化'),
            'vae_chunk_size': ('VAE chunk size（0: 無効）', 0, None, '3D VAEで有効。指定する場合は正の偶数'),
            'vae_disable_cache': ('3D VAE 内部キャッシュを無効化', False, None, '2D VAEでは使用しません。Latentキャッシュとは別'),
            'block_swap_h2d_only': ('高速なH2D-only block swap',
                                    True,
                                    None,
                                    'swap使用時のみ。gradient checkpointingが必要。VRAMとのトレードオフがあります'),
            'cuda_cudnn_benchmark': ('cuDNNの高速カーネルを自動選択',
                                     True,
                                     None,
                                     '同じ学習条件で実行方式を調整。初回は計測時間が発生し、ビット単位の一致は保証されません'),
            'cuda_allow_tf32': ('TF32を許可（任意）', False, None, 'FP32演算の精度を変更します。既定OFF。学習条件とは別の数値精度設定'),
            'compile_mode': ('Compile mode',
                             'default',
                             ['default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'],
                             '長時間学習ではmax-autotuneも比較可能。初回コンパイルと追加VRAMが必要'),
            'compile_dynamic': ('Compile dynamic shapes',
                                'auto',
                                ['auto', 'true', 'false'],
                                'bucket使用時はauto。WindowsでtrueにはVisual Studio C++コンパイラが必要'),
            'compile_cache_size_limit': ('Compile cache上限', 32, None, '解像度bucketごとの再コンパイルを抑制。compile有効時だけ適用'),
            'fused_adamw': ('Fused AdamW（AdamW選択時）',
                            True,
                            None,
                            'PyTorch AdamWのGPU演算を融合。AdamW8bit / CAME / Adafactorには適用しません')},
 'flow': {'timestep_sampling': ('Timestep sampling',
                                'shift',
                                ['sigma',
                                 'uniform',
                                 'sigmoid',
                                 'shift',
                                 'flux_shift',
                                 'truncated_logit_normal'],
                                'ノイズを加える時刻の分布。shiftはsigmoid分布にflow shiftを適用'),
          'weighting_scheme': ('Loss weighting',
                               'none',
                               ['none', 'sigma_sqrt', 'cosmap'],
                               'サンプリング分布とは別のloss重み付け'),
          'logit_mean': ('Logit mean', 0.0, None, ''),
          'logit_std': ('Logit std', 1.0, None, ''),
          'discrete_flow_shift': ('学習用 flow shift', 3.0, None, 'shift方式で有効。生成用shiftとは別'),
          'sigmoid_scale': ('Sigmoid scale', 1.0, None, ''),
          'exclude_patterns': ('除外パターン（1行1正規表現）', '', 'multiline', '例: .*mlp.*。Animaのモジュール名に完全一致'),
          'include_patterns': ('包含パターン（1行1正規表現）', '', 'multiline', '除外に対する例外。通常は空欄'),
          'network_reg_dims': ('モジュール別 rank', '', None, '例: .*self_attn.*=8,.*mlp.*=16'),
          'network_reg_lrs': ('モジュール別学習率', '', None, '例: blocks[.]0[.]cross_attn[.].*=2.5e-5。層別LRより優先'),
          'rank_dropout': ('Rank dropout', 0.0, None, ''),
          'module_dropout': ('Module dropout', 0.0, None, ''),
          'train_llm_adapter': ('LLM AdapterにもLoRAを適用', False, None, 'Qwen3本体の学習とは別。DiTと合わせて適用。既定OFF'),
          'self_attn_lr': ('Self-Attention LR', '', None, '画像パッチ同士の処理。空欄＝基本LR、0＝この系統を学習しない'),
          'cross_attn_lr': ('Cross-Attention LR', '', None, '画像とテキスト条件の対応。空欄＝基本LR、0＝この系統を学習しない'),
          'mlp_lr': ('MLP LR', '', None, '各ブロック内の特徴変換。空欄＝基本LR、0＝この系統を学習しない')},
 'samples': {'enabled': ('学習中にサンプル画像を生成', False, None, 'VRAM 12GB 向けの初期値は OFF。生成時にも VRAM を使用します'),
             'sample_every_n_epochs': ('何エポックごとに生成（0: 無効）', 1, None, ''),
             'sample_every_n_steps': ('何 steps ごとに生成（0: 無効）', 0, None, ''),
             'sample_at_first': ('学習前にも生成', False, None, ''),
             'width': ('生成画像の幅', 1024, None, '16 の倍数。VRAM が不足する場合は 768 / 512 へ'),
             'height': ('生成画像の高さ', 1024, None, '16 の倍数'),
             'infer_steps': ('生成ステップ数', 30, None, ''),
             'cfg': ('CFG scale', 3.0, None, '公式パイプラインの初期値'),
             'flow_shift': ('生成用 flow shift', 3.0, None, '学習用shiftとは別'),
             'sample_seed': ('生成 seed', 42, None, '同じ seed と prompt で学習前後を比較'),
             'negative_prompt': ('Negative prompt', '', None, '空欄可'),
             'prompts': ('プロンプト（1行1件）',
                         '1girl, solo, standing, outdoors, detailed eyes',
                         'multiline',
                         '学習対象のトリガーと特徴へ変更。--w / --h / --s / --l / --fs / --d / --nで行ごとに上書き')}}

ENV_DEFAULTS = {"python": "", "sd_scripts": ""}

def defaults(backend_name='sd-scripts', *, legacy=False):
    return {group: {key: copy.deepcopy(spec[1]) for key, spec in values.items()} for group, values in FIELDS.items()}


def migrate_project(project):
    if not isinstance(project,dict) or project.get('kind') != PROJECT_KIND:
        return
    settings=project.get('settings')
    if not isinstance(settings,dict):
        return
    training=settings.get('training',{})
    if isinstance(training,dict) and 'network_algo' in training:
        if training['network_algo'] != 'LoRA':
            raise ValueError(tr('このアプリはLoRAのみ対応しています。'))
        training.pop('network_algo')
    flow=settings.get('flow',{})
    if isinstance(flow,dict):
        flow.pop('mode_scale',None)
        if flow.get('train_llm_adapter'):
            raise ValueError(tr('LLM Adapter学習を含む旧プロジェクトは、この版では再開できません。新規プロジェクトでDiTのみの学習を設定してください。'))
    subsets=project.get('subsets',[])
    if isinstance(subsets,list):
        for sub in subsets:
            if isinstance(sub,dict):sub.setdefault('is_reg',False)


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


class ProjectStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def directory(self, project):
        pid = project['id']
        if not re.fullmatch(r'[a-f0-9]{32}', pid):
            raise ValueError(tr('プロジェクト ID が不正です'))
        return self.root / pid

    def create(self, name, source=None, *, preserve_output_name=False, model_settings=None, backend_name='sd-scripts'):
        name = name.strip()
        if not name or len(name) > 100:
            raise ValueError(tr('名前は 1〜100 文字で入力してください'))
        if any(p['name'].casefold() == name.casefold() for p in self.list()):
            raise ValueError(tr('同じ名前のプロジェクトがあります'))
        settings = copy.deepcopy(source['settings']) if source else defaults(backend_name)
        if source is None and model_settings is not None:
            # Keep the selected models while starting a fresh training plan.
            settings['model'] = {key: copy.deepcopy(model_settings.get(key, spec[1]))
                                 for key, spec in FIELDS['model'].items()}
        # Source paths are settings; run identity, state and results never carry over.
        settings['training'].update(resume='', network_weights='')
        if not preserve_output_name:
            settings['training']['output_name'] = 'anima_lora'
        project = {'kind': PROJECT_KIND, 'schema_version': SCHEMA_VERSION, 'id': uuid4().hex, 'name': name,
                   'created_at': datetime.now().astimezone().isoformat(), 'settings': settings,
                   'subsets': copy.deepcopy(source['subsets']) if source else [{'image_dir': '', 'num_repeats': 1, 'class_tokens': '', 'is_reg': False}]}
        self.save(project)
        return project

    def save(self, project):
        atomic_json(self.directory(project) / 'project.json', project)

    def remove(self, project):
        source = self.directory(project)
        if source.resolve() != source or source.resolve().parent != self.root or not source.is_dir():
            raise ValueError(tr('プロジェクトの保存先が不正です'))
        trash = self.root / '_trash'
        if trash.resolve() != trash:
            raise ValueError(tr('削除済みプロジェクトの保存先が不正です'))
        trash.mkdir(exist_ok=True)
        destination = trash / (project['id'] + '_' + uuid4().hex[:12])
        source.rename(destination)
        return destination

    def load(self, path):
        p = json.loads(Path(path).read_text(encoding='utf-8'))
        migrate_project(p)
        if not isinstance(p, dict) or (p.get('schema_version') != SCHEMA_VERSION or p.get('kind') != PROJECT_KIND):
            raise ValueError(tr('未対応のプロジェクト形式です'))
        if not isinstance(p.get('id'), str):
            raise ValueError(tr('プロジェクト ID が不正です'))
        self.directory(p)
        if not isinstance(p.get('name'), str) or not p['name'].strip() or len(p['name']) > 100 or not isinstance(p.get('subsets'), list) or not isinstance(p.get('settings'), dict):
            raise ValueError(tr('プロジェクト形式が不正です'))
        for sub in p['subsets']:
            if not isinstance(sub, dict) or set(sub) - {'image_dir', 'num_repeats', 'class_tokens','is_reg'}:
                raise ValueError(tr('画像サブセットの形式が不正です'))
            if not isinstance(sub.get('image_dir'), str) or type(sub.get('num_repeats')) is not int or not isinstance(sub.get('class_tokens'), str) or type(sub.get('is_reg',False)) is not bool:
                raise ValueError(tr('画像サブセットの型が不正です'))
        # Missing known fields get defaults; unknown fields are rejected, never silently launched.
        merged = defaults('sd-scripts', legacy=True)
        # A project saved before backend selection existed always used sd-scripts.
        for group, values in p['settings'].items():
            if group not in merged or not isinstance(values, dict):
                raise ValueError(tr('不明な設定: {p0}', p0=group))
            for key, value in values.items():
                if key not in merged[group]:
                    raise ValueError(tr('不明な設定: {p0}', p0=key))
                expected = type(merged[group][key])
                if type(value) is not expected and not (expected is float and type(value) is int):
                    raise ValueError(tr('設定の型が不正: {p0}', p0=key))
                if expected in (int, float) and not math.isfinite(value):
                    raise ValueError(tr('設定に非有限数は使えません: {p0}', p0=key))
                merged[group][key] = value
        p['settings'] = merged
        return p

    def list(self):
        result = []
        for path in self.root.glob('*/project.json'):
            # Corruption must be visible instead of silently hiding a project.
            result.append(self.load(path))
        return sorted(result, key=lambda p: p['name'].casefold())


def validate(project, env=None, files=False):
    s = project['settings']
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    for group, fields in FIELDS.items():
        for key, (_, default, choices, _) in fields.items():
            value = s[group][key]
            if type(default) in (int, float):
                require(type(value) in (int, float) and math.isfinite(value), tr('{p0}: 有限の数値が必要です', p0=key))
                if type(default) is int:
                    require(type(value) is int, tr('{p0}: 整数が必要です', p0=key))
            if isinstance(choices, list):
                require(value in choices, tr('{p0}: 選択肢が不正です', p0=key))
    if errors:
        return errors
    d, t, m, f = (s[k] for k in ('dataset', 'training', 'memory', 'flow'))
    is_sd_scripts = backend(s) == 'sd-scripts'
    for key in (('resolution', 'bucket_reso_steps', 'min_bucket_reso', 'max_bucket_reso') if is_sd_scripts else ('resolution',)):
        require(d[key] >= 16 and d[key] % 16 == 0, tr('{p0}: 16 以上の 16 の倍数にしてください', p0=key))
    if is_sd_scripts:
        require(d['min_bucket_reso'] <= d['max_bucket_reso'], tr('Bucket 最小値は最大値以下にしてください'))
        for key in ('min_bucket_reso', 'max_bucket_reso'):
            require(d[key] % d['bucket_reso_steps'] == 0, tr('{p0}: Bucket 刻みの倍数にしてください', p0=key))
    require(d['batch_size'] >= 1, tr('バッチサイズは 1 以上にしてください'))
    for key in ('qwen3_max_token_length', 't5_max_token_length'):
        require(1 <= s['model'][key] <= 32768, tr('{p0}: 1〜32768 にしてください', p0=key))
    for key in ('network_dim', 'network_alpha', 'learning_rate', 'gradient_accumulation_steps'):
        require(t[key] > 0, tr('{p0}: 0 より大きい値が必要です', p0=key))
    if t['weight_decay'].strip():
        try:
            decay = float(t['weight_decay'])
            require(math.isfinite(decay) and decay >= 0, tr('Weight decayは0以上の有限の数値にしてください。'))
        except ValueError:
            require(False, tr('Weight decayは0以上の有限の数値にしてください。'))
    if t['lr_scheduler'] == 'polynomial':
        require(t['lr_scheduler_power'] > 0, tr('Polynomial powerは0より大きい値にしてください。'))
    require(t['max_train_epochs'] >= 0, tr('{p0}: 0 以上にしてください', p0='max_train_epochs'))
    if t['max_train_steps'] == 0:
        require(t['max_train_epochs'] > 0, tr('{p0}: 0 より大きい値が必要です', p0='max_train_epochs'))
    if t['lr_scheduler'] == 'cosine_with_restarts':
        require(t['lr_scheduler_num_cycles'] > 0, tr('Restart周期数は1以上にしてください。'))
        if t['max_train_steps'] > 0:
            require(t['max_train_steps'] - t['lr_warmup_steps'] >= t['lr_scheduler_num_cycles'],
                    tr('再スタートにはwarmup後に周期数以上のSTEP数が必要です。'))
    for key in ('max_train_steps', 'lr_warmup_steps', 'save_every_n_steps', 'save_every_n_epochs', 'max_grad_norm', 'seed'):
        require(t[key] >= 0, tr('{p0}: 0 以上にしてください', p0=key))
    require(bool(re.fullmatch(r'[^<>:"/\\|?*\x00-\x1f]+', t['output_name'])) and t['output_name'] not in ('.', '..') and not t['output_name'].endswith((' ', '.')), tr('出力 LoRA 名に使用できない文字があります'))
    require(not re.fullmatch(r'(?i)(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\..*)?', t['output_name']), tr('Windows の予約名は出力名に使用できません'))
    require(d['caption_extension'].startswith('.') and not re.search(r'[/\\\s]', d['caption_extension']), tr('キャプション拡張子は .txt の形式で入力してください'))
    for group, key in (('dataset', 'caption_dropout_rate'), ('flow', 'rank_dropout'), ('flow', 'module_dropout')):
        require(0 <= s[group][key] <= 1, tr('{p0}: 0〜1 にしてください', p0=key))
    if is_sd_scripts:
        require(m['blocks_to_swap'] >= 0, tr('Block swap は 0 以上にしてください'))
        modes = sum([m['blocks_to_swap'] > 0, m['cpu_offload_checkpointing'], m['unsloth_offload_checkpointing']])
        require(modes <= 1, tr('Block swap / CPU offload / Unsloth はいずれか 1 つだけ選んでください'))
    require(not (m['cache_latents_to_disk'] and not m['cache_latents']), tr('ディスク Latent キャッシュには Latent キャッシュを ON にしてください'))
    require(not (m['cache_text_encoder_outputs_to_disk'] and not m['cache_text_encoder_outputs']), tr('ディスク TE キャッシュにはテキスト出力キャッシュを ON にしてください'))
    require(not (m['cache_latents'] and (d['color_aug'] or d['random_crop'])), tr('Latent キャッシュと色拡張 / ランダムクロップは併用できません'))
    require(not (m['cache_text_encoder_outputs'] and d['shuffle_caption']), tr('TE キャッシュとキャプションシャッフルは併用できません'))
    require(m['compile_cache_size_limit'] >= 1, tr('Compile cache上限は1以上にしてください'))
    require(m['max_data_loader_n_workers'] >= 0, tr('DataLoader workers は 0 以上にしてください'))
    require(m['num_cpu_threads_per_process'] >= 1, tr('CPU処理スレッド数は1以上の整数にしてください'))
    if is_sd_scripts:
        require(not (m['max_data_loader_n_workers'] == 0 and m['persistent_data_loader_workers']), tr('workers=0 の場合は worker 維持を OFF にしてください'))
        require(m['attn_mode'] != 'xformers' or m['split_attn'], tr('xformers には Split attention が必要です'))
    errors.extend(validate_optimization(s))
    errors.extend(validate_anima(s))
    require(0<=t['network_dropout']<=1,tr('Neuron dropoutは0〜1にしてください。'))
    require(t['scale_weight_norms']>=0,tr('LoRA weight norm上限は0以上にしてください。'))
    require(d['prior_loss_weight']>=0,tr('正則化画像のLoss重みは0以上にしてください。'))
    for key in ('logit_std', 'discrete_flow_shift', 'sigmoid_scale'):
        require(f[key] > 0, tr('{p0}: 0 より大きい値が必要です', p0=key))
    for key in ('exclude_patterns', 'include_patterns'):
        for pattern in f[key].splitlines():
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(tr('{p0}: 正規表現が不正: {p1}', p0=key, p1=exc))
    for key in ('network_reg_dims', 'network_reg_lrs'):
        for item in filter(None, f[key].split(',')):
            try:
                pattern, value = item.rsplit('=', 1)
                re.compile(pattern)
                number = int(value) if key.endswith('dims') else float(value)
                require(math.isfinite(number) and number > 0, tr('{p0}: 正数を指定してください', p0=key))
            except (ValueError, re.error):
                errors.append(tr('{p0}: pattern=値 の形式を確認してください', p0=key))
    if s['samples']['enabled']:
        require(any(line.strip() and not line.strip().startswith('#') for line in s['samples']['prompts'].splitlines()), tr('サンプルプロンプトを入力してください'))
        require(s['samples']['sample_every_n_epochs'] >= 0 and s['samples']['sample_every_n_steps'] >= 0 and (s['samples']['sample_every_n_epochs'] > 0 or s['samples']['sample_every_n_steps'] > 0), tr('サンプル生成間隔を steps または epoch で 1 以上にしてください'))
    sample = s['samples']
    for key in ('width', 'height'):
        require(sample[key] >= 16 and sample[key] % 16 == 0, tr('生成 {p0}: 16 以上の 16 の倍数にしてください', p0=key))
    require(sample['infer_steps'] >= 1, tr('生成ステップ数は 1 以上にしてください'))
    require(sample['cfg'] >= 0 and sample['flow_shift'] > 0 and sample['sample_seed'] >= 0, tr('生成 CFG / seed は 0 以上、生成 shift は正数にしてください'))
    require(not re.search(r'[\r\n]|--', sample['negative_prompt']), tr('共通 Negative prompt に改行や -- は使用できません'))
    require(any(not sub.get('is_reg',False) for sub in project['subsets']),tr('教師画像のサブセットが1つ以上必要です。正則化画像だけでは学習できません。'))
    require(bool(project['subsets']), tr('画像サブセットを 1 件以上追加してください'))
    for i, sub in enumerate(project['subsets'], 1):
        require(type(sub.get('is_reg',False)) is bool,tr('正則化画像の指定が不正です。'))
        require(type(sub.get('num_repeats')) is int and sub['num_repeats'] >= 1, tr('サブセット {p0}: repeat は 1 以上の整数です', p0=i))
        require(bool(sub.get('image_dir', '').strip()), tr('サブセット {p0}: 画像フォルダを指定してください', p0=i))
        if files and sub.get('image_dir'):
            path = Path(sub['image_dir'])
            require(path.is_absolute() and path.is_dir(), tr('画像フォルダが存在しません（絶対パス）: {p0}', p0=path))
            if path.is_dir():
                require(any(p.suffix.lower() in IMAGE_EXTENSIONS for p in path.iterdir() if p.is_file()), tr('画像がありません: {p0}（サブフォルダは自動再帰しません）', p0=path))
    require(m['vae_chunk_size'] == 0 or (m['vae_chunk_size'] > 0 and m['vae_chunk_size'] % 2 == 0), tr('VAE chunk sizeは0または正の偶数にしてください'))
    if files:
        env = selected_environment(env or ENV_DEFAULTS, s)
        require(Path(env['python']).is_file() and Path(env['python']).is_absolute(), tr('学習用 Python の実行ファイルを絶対パスで指定してください'))
        require(environment_valid(env), tr('選択した学習エンジンのPythonとソースを「かんたん導入」または「実行環境」で指定してください。'))
        for key in ('pretrained_model_name_or_path', 'qwen3', 'vae'):
            value = s['model'][key]
            path = Path(value)
            valid = path.is_file() or (key == 'qwen3' and path.is_dir() and (path / 'config.json').is_file())
            require(bool(value) and path.is_absolute() and valid, tr('{p0}: モデルの絶対パスを確認してください', p0=FIELDS['model'][key][0]))
        for key in ('t5_tokenizer_path', 'llm_adapter_path'):
            value = s['model'][key]
            require(not value or (Path(value).is_absolute() and (Path(value).is_file() if key == 'llm_adapter_path' else Path(value).is_dir())), tr('{p0}: パスが存在しません（絶対パス）', p0=key))
        for key in ('resume', 'network_weights'):
            value = t[key]
            require(not value or (Path(value).is_absolute() and (Path(value).is_file() if key == 'network_weights' else Path(value).is_dir())), tr('{p0}: パスが存在しません（絶対パス）', p0=key))
        from .model_check import model_errors
        errors.extend(model_errors(s))
    if t['resume']:
        require(not t['network_weights'], tr('state からの再開と初期 LoRA 重みは同時に指定しないでください'))
        if files:
            info = state_info(t['resume'])
            errors.extend(info['errors'])
            if not info['errors']:
                if t['max_train_steps']:
                    require(t['max_train_steps'] > info['step'], tr('総 STEP 上限は保存済み {p0} steps より大きくしてください（追加 STEP 数ではありません）', p0=info['step']))
                else:
                    if info.get('final'):
                        require(t['max_train_epochs'] > info['epoch'], tr('最終stateの追加学習では総Epoch上限を保存済みEpochより大きくしてください'))
                    else:
                        require(t['max_train_epochs'] >= info['epoch'], tr('再開時の総エポック数は保存済み epoch {p0} 以上にしてください（同じエポックの途中から再開可能）', p0=info['epoch']))
                errors.extend(resume_differences(project, info))
                if resume_limit_changed(project,info) and (is_schedulefree(t['optimizer_type']) or t['lr_scheduler']=='adafactor'):
                    errors.append(tr('このOptimizer / Schedulerではstateの総上限変更に未対応です。同じ計画で再開するか、LoRA重みから追加学習してください。'))
    return errors


def state_info(folder):
    folder = Path(folder).resolve()
    info = {'backend': 'sd-scripts', 'path': str(folder), 'step': 0, 'epoch': 0, 'errors': [], 'snapshot': None, 'final': False}
    try:
        metadata = json.loads((folder / 'train_state.json').read_text(encoding='utf-8'))
        if type(metadata['current_step']) is not int or metadata['current_step'] < 0 or type(metadata['current_epoch']) is not int or metadata['current_epoch'] < 0:
            raise ValueError(tr('step / epoch が不正です'))
        info.update(step=metadata['current_step'], epoch=metadata['current_epoch'])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        info['errors'].append(tr('state の進捗情報を読めません: {p0}', p0=exc))
    # Inspect structure only; do not deserialize pickle/optimizer files in the GUI.
    for label, patterns in [('LoRA model', ('model*.safetensors', 'pytorch_model*.bin')),
                            ('optimizer', ('optimizer*.bin',)), ('scheduler', ('scheduler*.bin',)),
                            (tr('乱数状態'), ('random_states_*.pkl',))]:
        try:
            found = any(p.is_file() and p.stat().st_size > 0 for pattern in patterns for p in folder.glob(pattern))
        except OSError:
            found = False
        if not found:
            info['errors'].append(tr('state の {p0} ファイルが不足 / 空です: {p1}', p0=label, p1=folder))
    snapshot = folder.parent.parent / 'project_snapshot.json'
    if snapshot.exists():
        try:
            saved = json.loads(snapshot.read_text(encoding='utf-8'))
            migrate_project(saved)
            if (saved.get('schema_version') != SCHEMA_VERSION or saved.get('kind') != PROJECT_KIND) or not isinstance(saved.get('settings'), dict) or not isinstance(saved.get('subsets'), list):
                raise ValueError(tr('未対応の snapshot 形式です'))
            for group, values in saved['settings'].items():
                if group not in FIELDS or not isinstance(values, dict) or any(key not in FIELDS[group] for key in values):
                    raise ValueError(tr('snapshot の設定形式が不正です'))
            info['snapshot'] = saved
            info['final'] = folder.name == saved['settings'].get('training', {}).get('output_name', 'anima_lora') + '-state'
        except (OSError, ValueError) as exc:
            info['errors'].append(tr('再開元の設定を読めません: {p0}', p0=exc))
    return info


def resume_differences(project, info):
    source = info.get('snapshot')
    if not source:
        return []
    differences = []
    groups = {'model': tuple(FIELDS['model']), 'dataset': tuple(FIELDS['dataset']), 'flow': tuple(FIELDS['flow']),
              'training': ('weight_decay','network_dim','network_alpha','optimizer_type','lr_scheduler','learning_rate','lr_warmup_steps','lr_scheduler_num_cycles','lr_scheduler_power','gradient_accumulation_steps','max_grad_norm','network_dropout','scale_weight_norms','seed','optimizer_args','lr_scheduler_args','lr_scheduler_min_lr_ratio','lr_scheduler_timescale','lr_decay_steps')}
    for group, keys in groups.items():
        for key in keys:
            before = source['settings'].get(group, {}).get(key, defaults()[group][key])
            if before != project['settings'][group][key]:
                differences.append(tr('state 再開の設定差: {p0}。元設定を復元するか、LoRA 重みから追加学習してください', p0=tr(FIELDS[group][key][0])))
    if source['subsets'] != project['subsets']:
        differences.append(tr('state 再開の設定差: 画像サブセット。元の画像パス・repeat・class tokens に戻してください'))
    return differences


def resume_limit_changed(project, info):
    source = info.get('snapshot')
    if not source:
        return True
    before, after = source['settings']['training'], project['settings']['training']
    if before.get('max_train_steps', 4000) or after['max_train_steps']:
        return before.get('max_train_steps', 4000) != after['max_train_steps']
    return before.get('max_train_epochs', 10) != after['max_train_epochs']


def discover_states(project_dir):
    result = []
    for metadata in Path(project_dir).glob('runs/*/models/*/train_state.json'):
        info = state_info(metadata.parent)
        info['modified'] = metadata.stat().st_mtime
        result.append(info)
    return sorted(result, key=lambda x: (x['modified'], x['step']), reverse=True)


def toml_value(value):
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (str, list)):
        return json.dumps(value, ensure_ascii=False)
    if not math.isfinite(value):
        raise ValueError(tr('TOML に非有限数は使用できません'))
    return str(value)


def dump_toml(values):
    return '\n'.join(f'{key} = {toml_value(value)}' for key, value in values.items()) + '\n'


def dataset_config(project):
    d = project['settings']['dataset']
    general_keys = ('caption_extension', 'enable_multiline_caption', 'shuffle_caption')
    dataset_keys = ('resolution', 'batch_size', 'enable_bucket', 'bucket_reso_steps', 'min_bucket_reso', 'max_bucket_reso', 'bucket_no_upscale')
    text = '[general]\n' + dump_toml({k: d[k] for k in general_keys})
    text += '\n[[datasets]]\n' + dump_toml({k: d[k] for k in dataset_keys})
    for sub in project['subsets']:
        text += '\n[[datasets.subsets]]\n' + dump_toml({**sub, 'is_reg':sub.get('is_reg',False), 'caption_dropout_rate': d['caption_dropout_rate'], 'color_aug': d['color_aug'], 'random_crop': d['random_crop']})
    return text


def training_config(project, run_dir):
    s = project['settings']
    run_dir = Path(run_dir).resolve()
    config = {'network_module': 'networks.lora_anima', 'network_train_unet_only': True,
              'mixed_precision': 'bf16', 'save_precision': 'bf16', 'save_model_as': 'safetensors',
              'loss_type': 'l2', 'text_encoder_batch_size': 1,
              'prior_loss_weight': s['dataset']['prior_loss_weight'],
              'output_dir': str(run_dir / 'models'), 'logging_dir': str(run_dir / 'tensorboard'),
              'log_with': 'tensorboard', 'dataset_config': str(run_dir / 'dataset.toml')}
    for group in ('model', 'training', 'memory'):
        config.update({k: v for k, v in s[group].items() if v != ''})
    weight_decay = config.pop('weight_decay', '')
    config.pop('network_algo', None)
    config['network_module'] = 'networks.lora_anima'
    if not config.get('vae_chunk_size'):
        config.pop('vae_chunk_size', None)
    if config['lr_scheduler'] != 'polynomial':
        config.pop('lr_scheduler_power',None)
    if config['lr_scheduler'] != 'cosine_with_restarts':
        config.pop('lr_scheduler_num_cycles',None)
    config.pop('keep_latest_state_only', None)
    config.pop('num_cpu_threads_per_process', None)  # accelerate launch option, not a trainer argument
    if not s['training']['resume']:
        config.pop('skip_until_initial_step', None)
    if config['max_train_steps'] > 0:
        config.pop('max_train_epochs')
    else:
        config.pop('max_train_steps')
    for key in ('save_every_n_steps', 'save_every_n_epochs'):
        if not config[key]:
            config.pop(key)
    config.pop('fused_adamw',None)
    if not config.get('blocks_to_swap'):
        config.pop('block_swap_h2d_only', None)
    elif config.get('block_swap_h2d_only'):
        config['gradient_checkpointing'] = True
    if not config.get('compile'):
        for key in ('compile_mode','compile_dynamic','compile_cache_size_limit'):
            config.pop(key,None)
    apply_optimization(config,s)
    if config['cpu_offload_checkpointing'] or config['unsloth_offload_checkpointing']:
        config['gradient_checkpointing'] = True
    f = s['flow']
    flow_keys = ('timestep_sampling', 'weighting_scheme', 'logit_mean', 'logit_std', 'discrete_flow_shift', 'sigmoid_scale')
    config.update({k: f[k] for k in flow_keys})
    args = [f'rank_dropout={f["rank_dropout"]}', f'module_dropout={f["module_dropout"]}']
    for key in ('exclude_patterns', 'include_patterns'):
        patterns = [line.strip() for line in f[key].splitlines() if line.strip()]
        if patterns:
            args.append(f'{key}={patterns!r}')
    args.extend(component_network_args(f))
    config['network_args'] = args
    if s['samples']['enabled']:
        config.update({k: s['samples'][k] for k in ('sample_every_n_epochs', 'sample_every_n_steps', 'sample_at_first')})
        for key in ('sample_every_n_epochs', 'sample_every_n_steps'):
            if not config[key]:
                config.pop(key)
        config['sample_prompts'] = str(run_dir / 'sample_prompts.txt')
    return config


def sample_prompts_text(settings):
    s = settings['samples']
    options = {'w': s['width'], 'h': s['height'], 's': s['infer_steps'], 'l': s['cfg'],
               'fs': s['flow_shift'], 'd': s['sample_seed']}
    if s['negative_prompt'].strip():
        options['n'] = s['negative_prompt'].strip()
    lines = []
    for line in s['prompts'].splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        suffix = ''.join(f' --{key} {value}' for key, value in options.items()
                         if not re.search(r'--' + key + r'(?:\s|$)', line))
        lines.append(line + suffix)
    return '\n'.join(lines) + '\n'


def command(env, run_dir, settings=None):
    env = selected_environment(env, settings or {})
    threads = settings.get('memory', {}).get('num_cpu_threads_per_process', 2) if settings else 2
    if type(threads) is not int or threads < 1:
        raise ValueError(tr('CPU処理スレッド数は1以上の整数にしてください'))
    return [env['python'], '-u', '-m', 'accelerate.commands.launch', '--num_processes', '1',
            '--num_machines', '1', '--mixed_precision', 'bf16', '--dynamo_backend', 'no',
            '--num_cpu_threads_per_process', str(threads), str(Path(__file__).resolve().with_name('resume_bridge.py')),
            str(Path(env['sd_scripts']).resolve()),
            '--state-policy', str(Path(run_dir).resolve() / 'state_policy.json'),
            '--config_file', str(Path(run_dir).resolve() / 'train.toml')]


def write_run(project, env, run_dir):
    env = selected_environment(env, project['settings'])
    errors = validate(project)
    if errors:
        raise ValueError('\n'.join(errors))
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'dataset.toml').write_text(dataset_config(project), encoding='utf-8')
    (run_dir / 'train.toml').write_text(dump_toml(training_config(project, run_dir)), encoding='utf-8')
    if project['settings']['samples']['enabled']:
        (run_dir / 'sample_prompts.txt').write_text(sample_prompts_text(project['settings']), encoding='utf-8')
    atomic_json(run_dir / 'project_snapshot.json', project)
    atomic_json(run_dir / 'environment_snapshot.json', env)
    info = state_info(project['settings']['training']['resume']) if project['settings']['training']['resume'] else None
    atomic_json(run_dir / 'state_policy.json', {
        'keep_latest_state_only': project['settings']['training']['keep_latest_state_only'],
        'recalculate_scheduler': bool(info and resume_limit_changed(project, info))})
    argv = command(env, run_dir, project['settings'])
    atomic_json(run_dir / 'command.json', argv)
    if project['settings']['training']['resume']:
        atomic_json(run_dir / 'resume_source.json', {key: value for key, value in info.items() if key != 'snapshot'})
    return argv


def caption_audit(folder, extension='.txt', multiline=True):
    """Read captions using sd-scripts naming rules; never edit dataset files."""
    if not extension.startswith('.') or re.search(r'[/\\\s]', extension):
        raise ValueError(tr('キャプション拡張子は .txt の形式で入力してください'))
    result = {key: [] for key in ('images', 'matched', 'readable', 'missing', 'empty', 'encoding_errors', 'read_errors', 'name_hints')}
    folder = Path(folder)
    if not folder.is_dir():
        return result
    result['images'] = sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: p.name.casefold())
    for image in result['images']:
        base = str(image.with_suffix(''))
        candidates = [Path(base + extension)]
        tokens = base.split('_')
        if len(tokens) >= 5:
            candidates.append(Path('_'.join(tokens[:-4]) + extension))
        caption = next((p for p in candidates if p.is_file()), None)
        if caption is None:
            result['missing'].append((image, candidates[0]))
            # These names look plausible but are not read by the backend.
            for hint in (Path(str(image) + extension), Path(base + extension + extension)):
                if hint.is_file():
                    result['name_hints'].append((hint, candidates[0]))
            continue
        result['matched'].append((image, caption))
        try:
            text = caption.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            result['encoding_errors'].append(caption)
            continue
        except OSError as exc:
            result['read_errors'].append((caption, str(exc)))
            continue
        used_text = text if multiline else next(iter(text.splitlines()), '')
        if not used_text.strip().strip('\ufeff').strip():
            result['empty'].append(caption)
        else:
            result['readable'].append((image, caption))
    return result


def dataset_report(project):
    lines, total = [], 0
    train_samples=reg_samples=0
    ext = project['settings']['dataset']['caption_extension']
    for sub in project['subsets']:
        folder = Path(sub['image_dir'])
        if not sub['image_dir'].strip():
            lines.append(tr('画像フォルダが未指定です。画像そのものを含むフォルダを選択してください。'))
            continue
        if not folder.is_dir():
            lines.append(tr('画像フォルダが存在しません（絶対パス）: {p0}', p0=folder))
            continue
        audit = caption_audit(folder, ext, project['settings']['dataset']['enable_multiline_caption'])
        images, missing = audit['images'], len(audit['missing'])
        count = len(images) * sub['num_repeats']
        total += count
        if sub.get('is_reg',False):reg_samples+=count
        else:train_samples+=count
        lines.append(tr('用途: 正則化画像') if sub.get('is_reg',False) else tr('用途: 教師画像'))
        lines.append(tr('{p0}: {p1} 枚 / caption なし {p2} 枚 / repeat 後 {p3} 枚', p0=folder, p1=len(images), p2=missing, p3=count))
        lines.append(tr('Caption 認識 {p0} / {p1} 件（同名ファイル {p2} 件 / 空の本文 {p3} 件 / UTF-8 エラー {p4} 件 / 読取エラー {p5} 件）',
                        p0=len(audit['readable']), p1=len(images), p2=len(audit['matched']), p3=len(audit['empty']), p4=len(audit['encoding_errors']), p5=len(audit['read_errors'])))
        lines.append(tr('検索設定: 拡張子 {p0} / 例: image.png → image{p0}（同じフォルダ）', p0=ext))
        for image, caption in audit['readable'][:3]:
            lines.append(tr('  認識: {p0} → {p1}', p0=image.name, p1=caption.name))
        for image, expected in audit['missing'][:10]:
            lines.append(tr('  Caption なし: {p0} / 探した名前: {p1}', p0=image.name, p1=expected.name))
        for actual, expected in audit['name_hints'][:10]:
            lines.append(tr('  名前を確認: {p0} は未対応です。期待する名前: {p1}', p0=actual.name, p1=expected.name))
        for caption in audit['empty'][:10]:
            lines.append(tr('  本文なし: {p0}（複数行OFFの場合は先頭行を使用）', p0=caption.name))
        for caption in audit['encoding_errors'][:10]:
            lines.append(tr('  UTF-8で保存し直してください: {p0}', p0=caption.name))
        for caption, error in audit['read_errors'][:10]:
            lines.append(tr('  読み取れません: {p0}: {p1}', p0=caption.name, p1=error))
        if not images:
            child_dirs = [p.name for p in folder.iterdir() if p.is_dir()][:10]
            if child_dirs:
                lines.append(tr('直下に画像がありません。子フォルダをサブセットとして追加してください: {p0}', p0=', '.join(child_dirs)))
    batch = project['settings']['dataset']['batch_size']
    accum = project['settings']['training']['gradient_accumulation_steps']
    if batch > 0 and accum > 0:
        lines.append(tr('合計 {p0} 枚相当 / 実効バッチ {p1} / 約 {p2} steps/epoch（単一 GPU）', p0=total, p1=batch * accum, p2=math.ceil(math.ceil(total / batch) / accum)))
    else:
        lines.append(tr('合計 {p0} 枚相当 / steps の概算には正のバッチサイズ・勾配累積が必要です', p0=total))
    lines.append(tr('教師画像 {p0} 枚相当 / 正則化画像 {p1} 枚相当。正則化の補充とbucket端数により実STEP数は変わります。',p0=train_samples,p1=reg_samples))
    lines.append(tr('caption がない画像には class tokens が使われます。画像フォルダは再帰走査しません。'))
    return '\n'.join(lines)


PROBE_CODE = '''import json, importlib.util
r={}
for m in ('torch','accelerate','safetensors','transformers','bitsandbytes','tensorboard','triton','xformers','flash_attn','pytorch_optimizer','lion_pytorch','prodigyopt','dadaptation','schedulefree'):
 r[m]=importlib.util.find_spec(m) is not None
if r['torch']:
 import torch
 r['torch_version']=torch.__version__
 r['cuda']=torch.cuda.is_available()
 if r['cuda']:
  r['gpu']=torch.cuda.get_device_name(0)
  r['vram_gb']=round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)
  r['bf16']=torch.cuda.is_bf16_supported()
  r['compute_capability']=list(torch.cuda.get_device_capability(0))
print(json.dumps(r))
'''


def probe(env):
    flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
    result = subprocess.run([env['python'], '-c', PROBE_CODE], cwd=source_dir(env), capture_output=True,
                            text=True, encoding='utf-8', errors='replace', timeout=90, creationflags=flags)
    if result.returncode:
        raise ValueError(result.stderr[-4000:] or tr('環境チェックに失敗しました'))
    info = json.loads(result.stdout.strip().splitlines()[-1])
    if info.get('triton') and info.get('cuda'):
        check = subprocess.run([env['python'], str(Path(__file__).with_name('triton_check.py'))],
                               cwd=source_dir(env), capture_output=True, text=True, encoding='utf-8',
                               errors='replace', timeout=180, creationflags=flags)
        info['triton_runtime'] = check.returncode == 0
        if check.returncode:
            info['triton_error'] = (check.stderr or check.stdout)[-2000:]
    return info


def probe_errors(info, settings):
    required = ['torch','accelerate','safetensors','transformers','tensorboard']
    dependency=DEPENDENCIES.get(settings['training']['optimizer_type'])
    if dependency:
        required.append(dependency)
    mode = settings['memory']['attn_mode']
    if mode != 'torch':
        required.append({'xformers':'xformers','flash':'flash_attn'}[mode])
    if settings['memory']['compile']:
        required.append('triton')
    errors = [tr('{p0} が学習用 Python にインストールされていません', p0=m) for m in required if not info.get(m)]
    if not info.get('cuda'):
        errors.append(tr('CUDA GPU を使用できません'))
    elif not info.get('bf16'):
        errors.append(tr('選択した GPU / PyTorch は bf16 に対応していません'))
    if settings['memory']['compile'] and info.get('triton_runtime') is False:
        errors.append(tr('torch.compile用のTriton実行テストに失敗しました'))
    return errors


class WindowsJob:
    """Own the complete Accelerate process tree, including training workers."""
    def __init__(self, process):
        import ctypes
        from ctypes import wintypes as w
        class BasicLimits(ctypes.Structure):
            _fields_ = [('ProcessTime', ctypes.c_int64), ('JobTime', ctypes.c_int64),
                        ('Flags', w.DWORD), ('MinWorkingSet', ctypes.c_size_t),
                        ('MaxWorkingSet', ctypes.c_size_t), ('ActiveProcessLimit', w.DWORD),
                        ('Affinity', ctypes.c_size_t), ('PriorityClass', w.DWORD), ('SchedulingClass', w.DWORD)]
        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in ('ReadOps', 'WriteOps', 'OtherOps', 'ReadBytes', 'WriteBytes', 'OtherBytes')]
        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('Basic', BasicLimits), ('Io', IoCounters),
                        ('ProcessMemory', ctypes.c_size_t), ('JobMemory', ctypes.c_size_t),
                        ('PeakProcessMemory', ctypes.c_size_t), ('PeakJobMemory', ctypes.c_size_t)]
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = w.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.kernel.SetInformationJobObject.restype = w.BOOL
        self.kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.kernel.AssignProcessToJobObject.restype = w.BOOL
        self.kernel.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.kernel.TerminateJobObject.restype = w.BOOL
        self.kernel.CloseHandle.argtypes = [w.HANDLE]
        self.kernel.CloseHandle.restype = w.BOOL
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            limits = ExtendedLimits()
            limits.Basic.Flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
        except Exception:
            self.close()
            raise

    def terminate(self):
        if self.handle:
            self.kernel.TerminateJobObject(self.handle, 1)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class Runner:
    """One child process tree, durable logs, and UI-independent event delivery."""
    def __init__(self, emit):
        self.emit = emit
        self.process = None
        self.thread = None
        self.stopping = False
        self.job = None

    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, argv, cwd, run_dir):
        if self.running:
            raise ValueError(tr('学習はすでに実行中です'))
        self.stopping = False
        self.thread = threading.Thread(target=self._run, args=(argv, cwd, Path(run_dir)), daemon=True)
        self.thread.start()

    def _run(self, argv, cwd, folder):
        started = datetime.now().astimezone().isoformat()
        status = {'status': 'running', 'started_at': started}
        try:
            atomic_json(folder / 'status.json', status)
            kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}
            env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
            with (folder / 'training.log').open('w', encoding='utf-8', buffering=1) as log:
                child = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                         text=True, encoding='utf-8', errors='replace', bufsize=1, **kwargs)
                try:
                    if os.name == 'nt':
                        self.job = WindowsJob(child)
                except Exception:
                    child.terminate()
                    child.wait()
                    child.stdout.close()
                    raise
                self.process = child
                if self.stopping:
                    self._terminate()
                for line in self.process.stdout:
                    log.write(line)
                    self.emit('log', line)
                code = self.process.wait()
                self.process.stdout.close()
            status.update(status='stopped' if self.stopping else ('completed' if code == 0 else 'failed'), returncode=code)
        except Exception as exc:
            status.update(status='failed', error=str(exc))
            self.emit('log', tr('\n起動エラー: {p0}\n', p0=exc))
        finally:
            if self.job:
                self.job.close()
                self.job = None
            status['finished_at'] = datetime.now().astimezone().isoformat()
            try:
                atomic_json(folder / 'status.json', status)
            finally:
                self.emit('done', status)

    def _terminate(self):
        p = self.process
        if p is None or p.poll() is not None:
            return
        if os.name == 'nt':
            if self.job:
                self.job.terminate()
        else:
            try:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def stop(self):
        self.stopping = True
        threading.Thread(target=self._terminate, daemon=True).start()

