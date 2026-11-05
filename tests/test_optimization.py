import copy,json
from pathlib import Path
import tempfile
import unittest
from anima_trainer.core import defaults,training_config,ProjectStore,validate
from anima_trainer.optimization import OPTIMIZERS,SCHEDULERS,parse_extra,validate_optimization


class OptimizationTests(unittest.TestCase):
    def test_catalog_has_supported_additions(self):
        self.assertTrue({'Lion','Prodigy','PagedAdamW8bit','DAdaptAdam','AdamWScheduleFree'}<=set(OPTIMIZERS))
        self.assertTrue({'cosine_with_min_lr','inverse_sqrt','warmup_stable_decay','adafactor'}<=set(SCHEDULERS))

    def test_reference_template_defaults(self):
        s=defaults();t=s['training'];d=s['dataset']
        self.assertEqual((t['network_dim'],t['network_alpha'],t['learning_rate']),(16,16,5e-5))
        self.assertEqual((t['optimizer_type'],t['lr_scheduler'],t['weight_decay']),('AdamW8bit','cosine','0.01'))
        self.assertEqual((t['max_train_steps'],t['max_train_epochs'],t['save_every_n_steps'],t['save_every_n_epochs']),(2000,15,100,0))
        self.assertEqual((d['resolution'],d['min_bucket_reso'],d['max_bucket_reso'],d['caption_dropout_rate']),(1024,512,1536,0.05))
        self.assertEqual((s['flow']['timestep_sampling'],s['flow']['discrete_flow_shift']),('shift',3.0))
        self.assertNotIn('mode_scale',s['flow']);self.assertNotIn('network_algo',t)

    def test_extra_arguments_are_literals_not_executed(self):
        self.assertEqual(parse_extra('betas=(0.9, 0.99)\neps=1e-8'),{'betas':(0.9,0.99),'eps':1e-8})
        for text in ('x=__import__("os").system("oops")','x=1\nx=2','x=1=2','x=1e999'):
            with self.subTest(text=text),self.assertRaises(ValueError):parse_extra(text)

    def test_extra_arguments_dont_override_dedicated_fields(self):
        s=defaults();s['training']['optimizer_args']='lr=1.0'
        self.assertTrue(validate_optimization(s))
        s['training']['optimizer_args']=''
        s['training']['lr_scheduler_args']='num_warmup_steps=50'
        self.assertTrue(validate_optimization(s))

    def test_schedulefree_validation_preserves_user_values(self):
        s=defaults();s['training']['optimizer_type']='AdamWScheduleFree'
        before=copy.deepcopy(s)
        self.assertTrue(validate_optimization(s));self.assertEqual(s,before)
        s['training']['lr_scheduler']='constant'
        self.assertEqual(validate_optimization(s),[])

    def test_scheduler_specific_validation(self):
        s=defaults();t=s['training']
        t['lr_scheduler']='inverse_sqrt'
        self.assertTrue(validate_optimization(s));t['lr_scheduler_timescale']=10
        self.assertEqual(validate_optimization(s),[])
        t.update(lr_scheduler='warmup_stable_decay',max_train_steps=100,lr_warmup_steps=10,lr_decay_steps=95)
        self.assertTrue(validate_optimization(s));t['lr_decay_steps']=40
        self.assertEqual(validate_optimization(s),[])

    def test_min_lr_decay_and_extra_args_export(self):
        s=defaults();t=s['training']
        t.update(lr_scheduler='cosine_with_min_lr',lr_scheduler_min_lr_ratio=0.1,optimizer_args='betas=(0.9, 0.99)')
        c=training_config({'settings':s},Path('run'))
        self.assertEqual(c['lr_scheduler_min_lr_ratio'],0.1)
        self.assertEqual(c['lr_scheduler_num_cycles'],1)
        self.assertIn('betas=(0.9, 0.99)',c['optimizer_args'])
        self.assertNotIn('lr_decay_steps',c)
        t.update(lr_scheduler='warmup_stable_decay',lr_decay_steps=50)
        self.assertEqual(training_config({'settings':s},Path('run'))['lr_decay_steps'],50)

    def test_adafactor_relative_schedule_is_explicit(self):
        s=defaults();t=s['training'];t['lr_scheduler']='adafactor'
        self.assertTrue(validate_optimization(s))
        t['optimizer_type']='Adafactor';self.assertEqual(validate_optimization(s),[])
        c=training_config({'settings':s},Path('run'))
        self.assertIn('relative_step=True',c['optimizer_args'])
        t['lr_scheduler']='cosine'
        self.assertIn('relative_step=False',training_config({'settings':s},Path('run'))['optimizer_args'])

    def test_legacy_project_keeps_its_training_values(self):
        with tempfile.TemporaryDirectory() as folder:
            store=ProjectStore(folder);p=store.create('old')
            p['settings']['training'].update(network_algo='LoRA',network_dim=8,network_alpha=1.0,max_train_steps=4000,max_train_epochs=10,lr_scheduler='constant')
            p['settings']['dataset']['resolution']=768
            p['settings']['flow']['mode_scale']=1.29
            for key in ('optimizer_args','lr_scheduler_args','lr_scheduler_min_lr_ratio','lr_scheduler_timescale','lr_decay_steps'):
                p['settings']['training'].pop(key)
            store.save(p)
            loaded=store.load(store.directory(p)/'project.json')
            self.assertEqual(loaded['settings']['training']['network_dim'],8)
            self.assertEqual(loaded['settings']['training']['lr_scheduler'],'constant')
            self.assertEqual(loaded['settings']['dataset']['resolution'],768)
            self.assertNotIn('network_algo',loaded['settings']['training'])
            self.assertNotIn('mode_scale',loaded['settings']['flow'])
