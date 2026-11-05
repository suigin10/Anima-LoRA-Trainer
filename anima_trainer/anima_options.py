"""Translate Anima component controls to the LoRA module's supported regex options."""
import math
from .i18n import tr

COMPONENTS = {
    'self_attn_lr': r'blocks\.[0-9]+\.self_attn\..*',
    'cross_attn_lr': r'blocks\.[0-9]+\.cross_attn\..*',
    'mlp_lr': r'blocks\.[0-9]+\.mlp\..*',
}


def component_values(flow):
    values={}
    for key in COMPONENTS:
        text=flow[key].strip()
        if not text:
            values[key]=None
            continue
        try:value=float(text)
        except ValueError:raise ValueError(tr('{p0}: 空欄または0以上の学習率を指定してください。',p0=key))
        if not math.isfinite(value) or value<0:
            raise ValueError(tr('{p0}: 空欄または0以上の学習率を指定してください。',p0=key))
        values[key]=value
    return values


def validate_anima(settings):
    errors=[]
    try:
        values=component_values(settings['flow'])
        if all(value==0 for value in values.values()):
            errors.append(tr('Self-Attention・Cross-Attention・MLPをすべて0にはできません。'))
        if settings['training']['optimizer_type'].endswith('ScheduleFree') and any(value is not None and value>0 for value in values.values()):
            errors.append(tr('この版のScheduleFreeでは層別LRは空欄または0にしてください。'))
        if settings['training']['lr_scheduler']=='adafactor' and any(value is not None and value>0 for value in values.values()):
            errors.append(tr('相対STEP方式のAdafactorでは層別LRは空欄または0にしてください。'))
    except ValueError as exc:errors.append(str(exc))
    if settings['flow']['train_llm_adapter']:
        errors.append(tr('LLM Adapterの学習は無効です。DiTのみを学習してください。'))
    return errors


def component_network_args(flow):
    values=component_values(flow)
    # Zero LR removes adapters at construction time through rank=0, so no optimizer
    # state or backward compute is spent on frozen modules. Include rules cannot
    # accidentally re-enable them. Manual ranks/LRs otherwise take precedence.
    frozen=[f'{COMPONENTS[key]}=0' for key,value in values.items() if value==0]
    dims=','.join(frozen+[flow['network_reg_dims']] if flow['network_reg_dims'].strip() else frozen)
    rates=[flow['network_reg_lrs']] if flow['network_reg_lrs'].strip() else []
    rates.extend(f'{COMPONENTS[key]}={value}' for key,value in values.items() if value is not None and value>0)
    args=[]
    if dims:args.append('network_reg_dims='+dims)
    if rates:args.append('network_reg_lrs='+','.join(rates))
    return args


def layer_preset(name,base_lr):
    values={key:'' for key in COMPONENTS}
    if name=='character':
        return dict(zip(COMPONENTS,('4e-5','2.5e-5','6e-5')))
    if name in COMPONENTS:
        values={key:str(base_lr) for key in COMPONENTS}
        values[name]='0'
    return values
