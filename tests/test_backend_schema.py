"""Integration with the pinned upstream schema/blueprint, without tensor/model imports.

Set ANIMA_TEST_SD_SCRIPTS to the installed or reviewed sd-scripts checkout.
"""
import argparse
import ast
import functools
import logging
import os
from pathlib import Path
import sys
import tomllib
import types
import unittest
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from anima_trainer.core import defaults, dataset_config, training_config


class BackendSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source=os.environ.get('ANIMA_TEST_SD_SCRIPTS')
        if not source:
            raise unittest.SkipTest('Set ANIMA_TEST_SD_SCRIPTS for real upstream schema integration')
        try:
            import voluptuous
        except ImportError:
            raise unittest.SkipTest('voluptuous is needed for upstream schema integration')
        path=Path(source)/'library/config_util.py'
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        # Execute the actual dataclasses, ConfigSanitizer and BlueprintGenerator,
        # omitting imports of model/dataset classes that are irrelevant to parsing.
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef)]
        module=types.ModuleType('_anima_upstream_schema_test')
        sys.modules[module.__name__]=module
        module.__dict__.update(argparse=argparse,functools=functools,logger=logging.getLogger(module.__name__),
            asdict=asdict,dataclass=dataclass,voluptuous=voluptuous,
            Dict=Dict,List=List,Optional=Optional,Sequence=Sequence,Tuple=Tuple,Union=Union,
            DEFAULT_MULTILINE_CAPTION_SEPARATOR='---===---')
        for name in ('Any','ExactSequence','MultipleInvalid','Object','Required','Schema'):
            module.__dict__[name]=getattr(voluptuous,name)
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(path),'exec'),module.__dict__)
        cls.backend=module
        cls.voluptuous=voluptuous

    def project(self,with_reg=False):
        p={'settings':defaults(),'subsets':[{'image_dir':'D:/teacher','num_repeats':2,'class_tokens':'','is_reg':False}]}
        if with_reg:p['subsets'].append({'image_dir':'D:/reg','num_repeats':1,'class_tokens':'person','is_reg':True})
        return p

    def test_generated_dataset_passes_real_upstream_schema(self):
        p=self.project()
        validator=self.backend.ConfigSanitizer(True,True,False,True)
        validated=validator.sanitize_user_config(tomllib.loads(dataset_config(p)))
        self.assertEqual(validated['datasets'][0]['resolution'],(1024,1024))

    def test_prior_loss_weight_reaches_dataset_blueprint_through_trainer_args(self):
        for weight in (0.0,0.35,1.0,2.0):
            with self.subTest(weight=weight):
                p=self.project(with_reg=True)
                p['settings']['dataset']['prior_loss_weight']=weight
                dataset=tomllib.loads(dataset_config(p))
                args=argparse.Namespace(**training_config(p,Path('run')))
                sanitizer=self.backend.ConfigSanitizer(True,True,False,True)
                blueprint=self.backend.BlueprintGenerator(sanitizer).generate(dataset,args)
                ds=blueprint.dataset_group.datasets[0]
                self.assertEqual(ds.params.prior_loss_weight,weight)
                self.assertFalse(ds.subsets[0].params.is_reg)
                self.assertTrue(ds.subsets[1].params.is_reg)

    def test_old_export_reproduces_the_reported_extra_key_error(self):
        config=tomllib.loads(dataset_config(self.project()))
        config['datasets'][0]['prior_loss_weight']=1.0
        sanitizer=self.backend.ConfigSanitizer(True,True,False,True)
        with self.assertLogs('_anima_upstream_schema_test',level='ERROR'):
            with self.assertRaises(self.voluptuous.MultipleInvalid) as error:
                sanitizer.sanitize_user_config(config)
        self.assertIn('prior_loss_weight',str(error.exception))
