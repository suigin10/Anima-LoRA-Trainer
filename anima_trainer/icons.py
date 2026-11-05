"""Small original PNG icons; GUI runtime needs only Tk, never Pillow."""
from pathlib import Path
import sys
import tkinter as tk
from tkinter import ttk
from .i18n import tr

ASSETS = Path(__file__).resolve().parents[1] / 'assets'


def load(app):
    app.studio_icon = tk.PhotoImage(master=app, file=str(ASSETS / 'anima-trainer-64.png'))
    app.iconphoto(True, app.studio_icon)
    app.studio_icons = {}
    for theme in ('dark', 'light'):
        for path in (ASSETS / 'icons').glob(f'*-{theme}.png'):
            app.studio_icons[path.stem] = tk.PhotoImage(master=app, file=str(path))


def decorate(app):
    buttons = {
        tr('▶ 学習を開始'): 'play', tr('■ 停止'): 'stop', tr('＋ 新規 / コピー'): 'plus',
        tr('削除'): 'trash', tr('読込'): 'import', tr('設定をエクスポート'): 'export',
        tr('上書き保存'): 'save', tr('別プロジェクトとして保存'): 'export', tr('フォルダ'): 'folder', tr('最新の学習フォルダ'): 'folder', tr('学習前チェック'): 'check',
        tr('実行設定のプレビュー'): 'settings', tr('画像・キャプションを確認'): 'folder',
        tr('実行環境'): 'settings', tr('かんたん導入'): 'install',
    }
    def visit(widget):
        if isinstance(widget, ttk.Button):
            text = str(widget.cget('text'))
            name = buttons.get(text)
            if name:
                theme = 'light' if widget.cget('style') == 'Header.TButton' or widget.cget('style') == 'Accent.TButton' else 'dark'
                widget.configure(image=app.studio_icons[f'{name}-{theme}'], compound='left',
                                 text=text.lstrip('▶■＋ '))
        for child in widget.winfo_children():
            visit(child)
    visit(app)
    for index, name in enumerate(('model','folder','chart','chip','camera','chart')):
        app.notebook.tab(index, image=app.studio_icons[name + '-dark'], compound='left')

