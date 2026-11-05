"""Optimizer/scheduler catalog and validated backend argument generation."""
import ast
import math
import re
from .i18n import tr

OPTIMIZERS = ['AdamW', 'AdamW8bit', 'PagedAdamW8bit', 'PagedAdamW32bit',
              'Lion', 'Lion8bit', 'PagedLion8bit', 'SGDNesterov', 'CAME',
              'Adafactor', 'Prodigy', 'DAdaptAdam', 'AdamWScheduleFree', 'RAdamScheduleFree']
SCHEDULERS = ['constant', 'constant_with_warmup', 'linear', 'cosine',
              'cosine_with_restarts', 'cosine_with_min_lr', 'polynomial',
              'inverse_sqrt', 'warmup_stable_decay', 'adafactor']
DEPENDENCIES = {'AdamW8bit':'bitsandbytes', 'PagedAdamW8bit':'bitsandbytes',
                'PagedAdamW32bit':'bitsandbytes', 'Lion8bit':'bitsandbytes',
                'PagedLion8bit':'bitsandbytes', 'Lion':'lion_pytorch',
                'CAME':'pytorch_optimizer', 'Prodigy':'prodigyopt',
                'DAdaptAdam':'dadaptation', 'AdamWScheduleFree':'schedulefree',
                'RAdamScheduleFree':'schedulefree'}


def is_schedulefree(name):
    return name.endswith('ScheduleFree')


def parse_extra(text, reserved=()):
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.count('=') != 1:
            raise ValueError(tr('追加引数は1行に key=値 を1つ指定してください。'))
        key, literal = [x.strip() for x in line.split('=',1)]
        if not re.fullmatch(r'[A-Za-z_]\w*',key) or key in result:
            raise ValueError(tr('追加引数の名前が不正または重複しています: {p0}',p0=key))
        if key in reserved:
            raise ValueError(tr('{p0} は専用の入力欄で設定してください。',p0=key))
        try:
            value = ast.literal_eval(literal)
        except (ValueError,SyntaxError):
            raise ValueError(tr('追加引数の値を確認してください: {p0}',p0=key))
        def valid(v):
            if isinstance(v,(list,tuple)):
                return all(valid(x) for x in v)
            return v is None or type(v) in (bool,str,int) or (type(v) is float and math.isfinite(v))
        if not valid(value):
            raise ValueError(tr('追加引数の値を確認してください: {p0}',p0=key))
        result[key] = value
    return result


OPT_RESERVED = {'lr','weight_decay','fused','relative_step','scale_parameter','warmup_init'}
SCHED_RESERVED = {'optimizer','num_warmup_steps','num_training_steps','num_cycles','power',
                  'timescale','min_lr_rate','min_lr_ratio','num_decay_steps','num_stable_steps'}


def validate_optimization(settings):
    t=settings['training']
    errors=[]
    for key,reserved in [('optimizer_args',OPT_RESERVED),('lr_scheduler_args',SCHED_RESERVED)]:
        try:parse_extra(t[key],reserved)
        except ValueError as exc:errors.append(str(exc))
    sf=is_schedulefree(t['optimizer_type'])
    scheduler=t['lr_scheduler']
    if sf and (scheduler!='constant' or t['lr_warmup_steps'] or t['lr_scheduler_args'].strip()):
        errors.append(tr('ScheduleFreeではSchedulerをconstant、Warmupを0、Scheduler追加引数を空欄にしてください。'))
    if scheduler=='adafactor' and (t['optimizer_type']!='Adafactor' or t['lr_warmup_steps'] or t['lr_scheduler_args'].strip()):
        errors.append(tr('adafactor SchedulerはAdafactor専用です。Warmupは0、Scheduler追加引数は空欄にしてください。'))
    if scheduler=='constant' and t['lr_warmup_steps']:
        errors.append(tr('constantではWarmupを0にしてください。Warmupを使う場合はconstant_with_warmupを選択してください。'))
    if not 0<=t['lr_scheduler_min_lr_ratio']<=1:
        errors.append(tr('最小学習率の比率は0〜1にしてください。'))
    if t['lr_scheduler_timescale']<0 or t['lr_decay_steps']<0:
        errors.append(tr('Timescaleと減衰STEP数は0以上にしてください。'))
    if scheduler=='inverse_sqrt' and t['lr_scheduler_timescale']==0 and t['lr_warmup_steps']==0:
        errors.append(tr('inverse_sqrtはWarmupまたはTimescaleを1以上にしてください。'))
    if scheduler=='warmup_stable_decay':
        if t['lr_decay_steps']<=0:
            errors.append(tr('warmup_stable_decayは減衰STEP数を1以上にしてください。'))
        if t['max_train_steps'] and t['lr_warmup_steps']+t['lr_decay_steps']>t['max_train_steps']:
            errors.append(tr('Warmupと減衰STEP数の合計は総STEP上限以下にしてください。'))
    if sf and settings['flow']['network_reg_lrs'].strip():
        errors.append(tr('この版のScheduleFreeではモジュール別学習率を空欄にしてください。'))
    return errors


def apply_optimization(config, settings):
    t=settings['training'];scheduler=t['lr_scheduler']
    config.pop('optimizer_args',None)
    config.pop('lr_scheduler_args',None)
    opt=parse_extra(t['optimizer_args'],OPT_RESERVED)
    sched=parse_extra(t['lr_scheduler_args'],SCHED_RESERVED)
    if settings['memory']['fused_adamw'] and t['optimizer_type']=='AdamW':
        opt['fused']=True
    if t['weight_decay'].strip():opt['weight_decay']=float(t['weight_decay'])
    if t['optimizer_type']=='CAME':config['optimizer_type']='pytorch_optimizer.CAME'
    if t['optimizer_type']=='Adafactor':
        opt.update(relative_step=scheduler=='adafactor',scale_parameter=False,warmup_init=False)
        config['max_grad_norm']=0.0
    if opt:config['optimizer_args']=[f'{k}={v!r}' for k,v in opt.items()]
    for key in ('lr_scheduler_min_lr_ratio','lr_scheduler_timescale','lr_decay_steps'):
        config.pop(key,None)
    if scheduler in ('cosine_with_min_lr','warmup_stable_decay'):
        config['lr_scheduler_min_lr_ratio']=t['lr_scheduler_min_lr_ratio']
        config['lr_scheduler_num_cycles']=1
    if scheduler=='inverse_sqrt' and t['lr_scheduler_timescale']:
        config['lr_scheduler_timescale']=t['lr_scheduler_timescale']
    if scheduler=='warmup_stable_decay':config['lr_decay_steps']=t['lr_decay_steps']
    if scheduler=='polynomial':sched.setdefault('lr_end',0.0)
    if sched:config['lr_scheduler_args']=[f'{k}={v!r}' for k,v in sched.items()]
    return config


def optimizer_hint(name):
    if is_schedulefree(name):
        return tr('ScheduleFree: Schedulerはconstant、Warmupは0。学習率は自動変更しません。')
    if name in ('Prodigy','DAdaptAdam'):
        return tr('自動適応型Optimizer: LRはAdamWと意味が異なります。通常1.0付近から検討します。入力値は変更しません。')
    if name=='Adafactor':
        return tr('通常は固定LR。adafactor Schedulerを選ぶと相対STEP方式。勾配クリップは出力時0。')
    return tr('Optimizerを変更しても学習率は維持します。追加引数は高度な設定で指定できます。')
