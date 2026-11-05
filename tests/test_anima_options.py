import copy,json
from pathlib import Path
import tempfile,tomllib,unittest
from anima_trainer.core import defaults,training_config,dataset_config,validate,ProjectStore
from anima_trainer.anima_options import component_network_args,validate_anima,layer_preset


class AnimaOptionsTests(unittest.TestCase):
    def setUp(self):
        self.s=defaults()
        self.p={'settings':self.s,'subsets':[{'image_dir':'D:/teacher','num_repeats':1,'class_tokens':'','is_reg':False}]}

    def test_layer_rates_export_as_lora_network_regexes(self):
        self.s['flow'].update(self_attn_lr='4e-5',cross_attn_lr='2.5e-5',mlp_lr='6e-5')
        c=training_config(self.p,Path('run'))
        self.assertNotIn('self_attn_lr',c)  # full-finetune CLI flags are not LoRA controls
        args=c['network_args']
        rates=next(x for x in args if x.startswith('network_reg_lrs='))
        self.assertIn('self_attn',rates);self.assertIn('cross_attn',rates);self.assertIn('mlp',rates)
        self.assertNotIn('train_llm_adapter=True',args)
        self.assertTrue(c['network_train_unet_only'])

    def test_zero_lr_removes_lora_modules_before_optimizer_creation(self):
        f=self.s['flow'];f['self_attn_lr']='0';f['network_reg_dims']='.*self_attn.*=8'
        arg=next(x for x in component_network_args(f) if x.startswith('network_reg_dims='))
        self.assertIn('self_attn',arg)
        self.assertTrue(arg.split('=',1)[1].split(',')[0].endswith('=0'))

    def test_manual_per_module_rate_has_priority_over_component_rate(self):
        f=self.s['flow'];f.update(cross_attn_lr='2.5e-5',network_reg_lrs='blocks.0.cross_attn.*=1e-5')
        rates=next(x for x in component_network_args(f) if x.startswith('network_reg_lrs='))
        self.assertTrue(rates.startswith('network_reg_lrs=blocks.0.cross_attn.*=1e-5,'))

    def test_blank_inherits_without_changing_any_rate(self):
        before=copy.deepcopy(self.s)
        self.assertEqual(component_network_args(self.s['flow']),[])
        self.assertEqual(self.s,before)

    def test_invalid_or_all_frozen_rates_are_rejected(self):
        f=self.s['flow']
        for text in ('-1','nan','inf','abc'):
            f['self_attn_lr']=text
            self.assertTrue(validate_anima(self.s))
        f.update(self_attn_lr='0',cross_attn_lr='0',mlp_lr='0')
        self.assertTrue(validate_anima(self.s))

    def test_adapter_learning_cannot_be_enabled(self):
        self.s['flow']['train_llm_adapter']=True
        self.assertTrue(validate_anima(self.s))
        self.assertNotIn('train_llm_adapter=True',training_config(self.p,Path('run'))['network_args'])

    def test_relative_adafactor_and_schedulefree_do_not_silently_ignore_rates(self):
        self.s['flow']['mlp_lr']='6e-5'
        self.s['training']['optimizer_type']='AdamWScheduleFree'
        self.assertTrue(validate_anima(self.s))
        self.s['training'].update(optimizer_type='Adafactor',lr_scheduler='adafactor')
        self.assertTrue(validate_anima(self.s))

    def test_regularization_subsets_and_loss_weight_reach_dataset(self):
        self.p['subsets'].append({'image_dir':'D:/regularization','num_repeats':1,'class_tokens':'person','is_reg':True})
        self.s['dataset']['prior_loss_weight']=0.5
        d=tomllib.loads(dataset_config(self.p))['datasets'][0]
        self.assertNotIn('prior_loss_weight',d)
        self.assertFalse(d['subsets'][0]['is_reg']);self.assertTrue(d['subsets'][1]['is_reg'])
        self.assertEqual(training_config(self.p,Path('run'))['prior_loss_weight'],0.5)

    def test_regularization_images_alone_are_rejected(self):
        self.p['subsets'][0]['is_reg']=True
        self.assertTrue(validate(self.p))

    def test_dropout_and_norm_options_export_without_default_changes(self):
        self.s['training'].update(network_dropout=0.1,scale_weight_norms=1.0)
        c=training_config(self.p,Path('run'))
        self.assertEqual(c['network_dropout'],0.1);self.assertEqual(c['scale_weight_norms'],1)
        self.assertEqual(self.s['training']['learning_rate'],5e-5)

    def test_old_subsets_default_to_teacher_and_existing_values_survive(self):
        with tempfile.TemporaryDirectory() as folder:
            store=ProjectStore(folder);p=store.create('old');p['settings']['dataset']['resolution']=768
            for sub in p['subsets']:sub.pop('is_reg',None)
            for k in ('self_attn_lr','cross_attn_lr','mlp_lr'):p['settings']['flow'].pop(k)
            store.save(p);loaded=store.load(store.directory(p)/'project.json')
            self.assertFalse(loaded['subsets'][0]['is_reg'])
            self.assertEqual(loaded['settings']['flow']['self_attn_lr'],'')
            self.assertEqual(loaded['settings']['dataset']['resolution'],768)

    def test_preset_only_returns_component_fields(self):
        self.assertEqual(layer_preset('character',1e-4),{'self_attn_lr':'4e-5','cross_attn_lr':'2.5e-5','mlp_lr':'6e-5'})
        self.assertEqual(layer_preset('self_attn_lr',1e-4)['self_attn_lr'],'0')
        self.assertEqual(layer_preset('base',1e-4),{'self_attn_lr':'','cross_attn_lr':'','mlp_lr':''})

    def test_old_adapter_training_project_cannot_resume_as_dit_only(self):
        with tempfile.TemporaryDirectory() as folder:
            store=ProjectStore(folder);p=store.create('old adapter run')
            p['settings']['flow']['train_llm_adapter']=True
            store.save(p)
            with self.assertRaises(ValueError):store.load(store.directory(p)/'project.json')
