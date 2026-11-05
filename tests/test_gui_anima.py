from pathlib import Path
import tempfile
import tkinter as tk
from tkinter import ttk
import unittest
from anima_trainer.gui import App, ProjectDialog, ResumeDialog


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.app=App(self.temp.name)
        self.addCleanup(self.app.destroy)
        self.app.withdraw()
        self.errors=[]
        self.app.report_callback_exception=lambda *e:self.errors.append(e)

    def test_speed_button_preserves_hyperparameters(self):
        before=self.app.collect()
        self.app.apply_speed_settings()
        after=self.app.collect()
        for group in ('model','dataset','training','flow','samples'):
            self.assertEqual(before['settings'][group],after['settings'][group])
        self.assertEqual(self.errors,[])

    def test_ui_is_blue_and_bilingual(self):
        self.assertEqual(ttk.Style(self.app).lookup('Accent.TButton','background'),'#2563b8')
        self.assertEqual(len(self.app.notebook.tabs()),6)
        self.app.language_choice.set('English');self.app.change_language();self.app.update()
        self.assertIn('Model',self.app.notebook.tab(0,'text'))
        self.assertEqual(self.errors,[])

    def test_project_copy_save_restart_and_dialogs(self):
        clone=self.app.store.create('copy',self.app.collect())
        self.app.refresh_projects(clone['id']);self.app.update()
        self.assertTrue(self.app.save())
        for factory in (ProjectDialog,ResumeDialog):
            dialog=factory(self.app);self.app.update();dialog.destroy()
        self.app.environment();self.app.update()
        for w in self.app.winfo_children():
            if isinstance(w,tk.Toplevel):w.destroy()
        self.app.setup_dialog();self.app.update()
        for w in self.app.winfo_children():
            if isinstance(w,tk.Toplevel):w.destroy()
        reopened=App(self.temp.name)
        try:self.assertEqual(reopened.project['id'],clone['id'])
        finally:reopened.destroy()
        self.assertEqual(self.errors,[])

    def test_model_path_buttons_are_equal_and_separated(self):
        self.app.deiconify();self.app.update()
        buttons=self.app.path_buttons[('model','qwen3')]
        self.assertEqual(len(buttons),2)
        self.assertEqual(buttons[0].winfo_width(),buttons[1].winfo_width())
        self.assertEqual(buttons[0].winfo_height(),buttons[1].winfo_height())
        self.assertGreater(buttons[1].winfo_x(),buttons[0].winfo_x()+buttons[0].winfo_width())

    def test_flow_is_folded_into_training_and_values_survive_collapse(self):
        self.assertNotIn('flow',self.app.pages)
        advanced=self.app.disclosures['training']
        self.assertFalse(advanced.opened.get())
        before=self.app.collect()
        advanced.toggle();self.app.update();advanced.toggle();self.app.update()
        self.assertEqual(self.app.collect(),before)
        self.assertIn(('flow','timestep_sampling'),self.app.variables)
        self.assertNotIn(('flow','mode_scale'),self.app.variables)

    def test_optimizer_change_does_not_change_learning_rate(self):
        original=self.app.variables[('training','learning_rate')].get()
        self.app.variables[('training','optimizer_type')].set('Prodigy')
        self.app.variables[('training','optimizer_type')].set('AdamWScheduleFree')
        self.assertEqual(self.app.variables[('training','learning_rate')].get(),original)
        self.assertEqual(tuple(self.app.field_widgets[('training','lr_scheduler')].cget('values')),('constant',))
        self.assertEqual(self.errors,[])

    def test_layer_preset_preserves_main_training_parameters(self):
        before=self.app.collect()
        label=next(label for label,code in self.app.layer_preset_codes.items() if code=='character')
        self.app.layer_preset_choice.set(label);self.app.apply_layer_preset()
        after=self.app.collect()
        for group in ('model','dataset','training','memory','samples'):
            self.assertEqual(before['settings'][group],after['settings'][group])
        self.assertEqual(after['settings']['flow']['self_attn_lr'],'4e-5')
        self.assertNotIn(('flow','train_llm_adapter'),self.app.variables)

    def test_regularization_checkbox_survives_project_save(self):
        row=self.app.subset_rows[0];row[4].set(True)
        self.assertTrue(self.app.collect()['subsets'][0]['is_reg'])
        self.assertTrue(self.app.save())
        reopened=App(self.app.data_dir)
        try:self.assertTrue(reopened.collect()['subsets'][0]['is_reg'])
        finally:reopened.destroy()

