import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import tomllib
import types
import unittest
from unittest.mock import patch

from anima_trainer.core import ProjectStore, defaults, validate, training_config, dataset_config, dump_toml, write_run, resume_differences
from anima_trainer.model_check import header, model_errors
from anima_trainer.cache_paths import install_cache_paths
from anima_trainer.backends import environment_valid
from anima_trainer.installer import install, extract_source


def tensor_file(path, values):
    data = json.dumps(values).encode()
    path.write_bytes(struct.pack('<Q', len(data)) + data + b'00')


class AnimaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.store = ProjectStore(self.root/'projects')
        self.project = self.store.create('test')
        self.project['subsets'][0]['image_dir'] = str(self.root)

    def test_defaults_and_project_roundtrip(self):
        p = self.store.load(self.store.directory(self.project)/'project.json')
        self.assertEqual(p['settings'],defaults())
        self.assertEqual(p['kind'],'AnimaLoRATrainer')
        self.assertEqual(validate(self.project),[])
        s=p['settings']
        self.assertEqual((s['training']['network_dim'],s['training']['network_alpha']),(16,16))
        self.assertEqual((s['training']['max_train_steps'],s['training']['max_train_epochs'],s['dataset']['resolution']),(2000,15,1024))

    def test_moss_project_is_rejected(self):
        p=copy.deepcopy(self.project);p.pop('kind')
        path=self.root/'moss.json';path.write_text(json.dumps(p),encoding='utf-8')
        with self.assertRaises(ValueError):self.store.load(path)

    def test_only_anima_trainer_arguments_exported(self):
        config=training_config(self.project,self.root/'run')
        parsed=tomllib.loads(dump_toml(config))
        self.assertEqual(parsed['network_module'],'networks.lora_anima')
        self.assertTrue(parsed['network_train_unet_only'])
        self.assertNotIn('network_algo',parsed)
        self.assertNotIn('fused_adamw',parsed)
        self.assertNotIn('block_swap_h2d_only',parsed)
        self.assertNotIn('compile_mode',parsed)
        self.assertFalse(any('kohaku' in k or 'moss' in k or 'convrot' in k for k in parsed))
        tomllib.loads(dataset_config(self.project))

    def test_adapter_stays_frozen_and_patterns_export(self):
        f=self.project['settings']['flow']
        f.update(exclude_patterns='.*mlp.*\n.*final_layer.*',network_reg_lrs='.*cross_attn.*=5e-5')
        c=training_config(self.project,self.root)
        self.assertNotIn('train_llm_adapter=True',c['network_args'])
        self.assertTrue(any('exclude_patterns=' in x for x in c['network_args']))
        self.assertIn('network_reg_lrs=.*cross_attn.*=5e-5',c['network_args'])

    def test_cache_conflicts_and_whole_caption_dropout(self):
        s=self.project['settings']
        s['dataset']['caption_dropout_rate']=0.2
        self.assertEqual(validate(self.project),[])
        s['dataset']['shuffle_caption']=True
        self.assertTrue(validate(self.project))
        s['dataset']['shuffle_caption']=False;s['dataset']['random_crop']=True
        self.assertTrue(validate(self.project))

    def test_swap_and_offload_exclusive(self):
        m=self.project['settings']['memory']
        m.update(blocks_to_swap=4,cpu_offload_checkpointing=True)
        self.assertTrue(validate(self.project))
        m['cpu_offload_checkpointing']=False;m['gradient_checkpointing']=False
        c=training_config(self.project,self.root)
        self.assertTrue(c['block_swap_h2d_only'])
        self.assertTrue(c['gradient_checkpointing'])

    def test_fused_adamw_applies_only_to_adamw(self):
        t=self.project['settings']['training']
        self.assertNotIn('fused=True',training_config(self.project,self.root).get('optimizer_args',[]))
        t['optimizer_type']='AdamW'
        self.assertIn('fused=True',training_config(self.project,self.root)['optimizer_args'])
        t['optimizer_type']='Adafactor'
        self.assertNotIn('fused=True',training_config(self.project,self.root)['optimizer_args'])

    def test_steps_epochs_and_compile_options(self):
        s=self.project['settings'];s['training']['max_train_steps']=0
        s['memory']['compile']=True
        c=training_config(self.project,self.root)
        self.assertNotIn('max_train_steps',c)
        self.assertEqual(c['max_train_epochs'],15)
        self.assertEqual(c['compile_cache_size_limit'],32)

    def test_write_run_writes_auditable_command(self):
        folder=self.root/'run'
        argv=write_run(self.project,{'python':sys.executable,'sd_scripts':str(self.root)},folder)
        self.assertEqual(json.loads((folder/'command.json').read_text()),argv)
        self.assertIn('resume_bridge.py',' '.join(argv))
        self.assertEqual(tomllib.loads((folder/'train.toml').read_text())['network_module'],'networks.lora_anima')
        self.assertTrue((folder/'state_policy.json').is_file())

    def test_resume_detects_rank_and_model_change(self):
        info={'snapshot':copy.deepcopy(self.project),'backend':'sd-scripts'}
        self.project['settings']['training']['network_dim']=32
        self.assertTrue(resume_differences(self.project,info))
        self.project['settings']['training']['network_dim']=16
        self.project['settings']['model']['qwen3']='other'
        self.assertTrue(resume_differences(self.project,info))

    def test_header_bounds(self):
        p=self.root/'bad.safetensors';p.write_bytes(struct.pack('<Q',2**40))
        with self.assertRaises(ValueError):header(p)

    def test_anima_29b_is_explicitly_rejected(self):
        values={f'blocks.{i}.self_attn.q_proj.weight':{'dtype':'BF16'} for i in range(40)}
        values['blocks.0.cross_attn.q_proj.weight']={'dtype':'BF16'}
        values['llm_adapter.out_proj.weight']={'dtype':'BF16'}
        p=self.root/'anima-2.9B.safetensors';tensor_file(p,values)
        s=self.project['settings'];s['model']['pretrained_model_name_or_path']=str(p)
        self.assertTrue(any('2.9B' in e for e in model_errors(s)))

    def test_model_architecture_quantization_adapter_and_swap(self):
        values={f'net.blocks.{i}.self_attn.q_proj.weight':{'dtype':'BF16'} for i in range(28)}
        values['net.blocks.0.cross_attn.q_proj.weight']={'dtype':'BF16'}
        values['net.llm_adapter.out_proj.weight']={'dtype':'BF16'}
        p=self.root/'anima.safetensors';tensor_file(p,values)
        s=self.project['settings'];s['model']['pretrained_model_name_or_path']=str(p)
        self.assertEqual(model_errors(s),[])
        s['memory']['blocks_to_swap']=27
        self.assertTrue(model_errors(s));s['memory']['blocks_to_swap']=0
        values.pop('net.llm_adapter.out_proj.weight');tensor_file(p,values)
        self.assertTrue(model_errors(s))
        adapter=self.root/'adapter.safetensors';tensor_file(adapter,{'llm_adapter.out_proj.weight':{'dtype':'BF16'}})
        s['model']['llm_adapter_path']=str(adapter)
        self.assertEqual(model_errors(s),[])
        values['net.blocks.0.self_attn.q_proj.weight']['dtype']='F8_E4M3';tensor_file(p,values)
        self.assertTrue(model_errors(s))

    def test_project_cache_changes_with_caption_model_and_resolution(self):
        class TE:
            def get_outputs_npz_path(self,p):return str(Path(p).with_suffix('.text.npz'))
        class Latent:
            def get_latents_npz_path(self,p,size):return str(Path(p).with_suffix(''))+f'_{size[0]:04d}x{size[1]:04d}_anima.npz'
        image=self.root/'image.png';image.write_bytes(b'image')
        caption=image.with_suffix('.txt');caption.write_text('first')
        strategy=types.SimpleNamespace(AnimaTextEncoderOutputsCachingStrategy=TE,AnimaLatentsCachingStrategy=Latent)
        install_cache_paths(strategy,self.project,self.root/'cache')
        first=TE().get_outputs_npz_path(str(image))
        self.assertTrue(Path(first).is_relative_to(self.root/'cache'))
        self.assertEqual(first,TE().get_outputs_npz_path(str(image)))
        caption.write_text('second')
        self.assertNotEqual(first,TE().get_outputs_npz_path(str(image)))
        latent=Latent().get_latents_npz_path(str(image),(1024,768))
        self.assertIn('_1024x0768_',Path(latent).name)
        self.assertFalse((self.root/'image.text.npz').exists())

    def test_installer_rejects_foreign_environment(self):
        (self.root/'anima-env').mkdir()
        with self.assertRaises(ValueError):install(self.root,self.root/'data')

    def test_archive_cannot_escape_destination(self):
        from zipfile import ZipFile
        archive=self.root/'bad.zip'
        with ZipFile(archive,'w') as z:z.writestr('root/../../outside','bad')
        with self.assertRaises(ValueError):extract_source(archive,self.root/'source')


if __name__=='__main__':unittest.main()
