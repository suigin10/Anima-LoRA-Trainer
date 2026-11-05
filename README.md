# Anima LoRA Trainer

Windows向けのAnima専用LoRA学習GUI。MOSS-LoRA-Trainerの画面構成・プロジェクト管理・ログ・再開操作を引き継ぎ、ブルー基調に変更しました。GUIはPython標準のtkinterだけで動作します。

## ダウンロード

- [v0.3.1 通常版ZIP](https://github.com/suigin10/Anima-LoRA-Trainer/raw/refs/heads/main/dist/Anima-LoRA-Trainer-v0.3.1.zip)
- [v0.3.1 修正パッチZIP](https://github.com/suigin10/Anima-LoRA-Trainer/raw/refs/heads/main/dist/Anima-LoRA-Trainer-v0.3.1-fix.zip)（v0.3.0導入済みの場合）
- [通常版SHA256](dist/Anima-LoRA-Trainer-v0.3.1.zip.sha256) / [修正パッチSHA256](dist/Anima-LoRA-Trainer-v0.3.1-fix.zip.sha256)

## 起動

1. ZIPをローカルの書き込み可能なフォルダに展開します。
2. Python 3.12（64bit、tkinterを含む）を用意し、`Start Anima LoRA Trainer.bat`を起動します。3.11も利用できます。
3. 右上の「かんたん導入」で専用環境を導入します。ダウンロードにはインターネット接続が必要です。
4. 「01 モデル」でAnima DiT、Qwen3-0.6B、Qwen-Image VAEを指定します。
5. 「02 データセット」で画像フォルダとrepeatsを指定します。画像と同名の`.txt`をキャプションとして使います。
6. 「学習前チェック」→「学習を開始」。出力は「最新の学習フォルダ」から開けます。

コマンドからの起動: `python launch.py`。GPU学習にはCUDA / bf16対応のNVIDIA GPUが必要です。GUI起動時に学習パッケージはインストールしません。

## v0.3.1の修正

- `prior_loss_weight`をdataset.tomlへ書き出して学習が停止する不具合を修正。sd-scriptsの仕様に合わせtrain.tomlへ出力します。
- 実際のsd-scripts ConfigSanitizer / BlueprintGeneratorを使った検証を追加。正則化Loss重みがデータセットへ反映されることも確認します。
- 導入済みの環境・モデル・データセット・プロジェクトの再作成は不要です。アプリ本体を更新し、通常の「学習を開始」で新しい実行設定を生成してください。

## v0.3.0の変更

- 学習対象をDiTのLoRAに固定。Qwen3、LLM Adapter、VAEは学習しません。
- 03の高度な設定にSelf-Attention / Cross-Attention / MLPの個別LRを追加。空欄で基本LR、0でその系統を無効化。
- 比較用の層別LR設定、正則化画像の指定、正則化画像のLoss重み、Neuron dropout、LoRA weight norm上限を追加。
- 新規初期値を基本LR 0.00005、1024px、2000 STEP、100 STEPごとの保存へ変更。50〜100枚・batch 1・勾配累積1で比較を始めるための値です。
- 初期値の由来を説明するGUI内の文章を削除。

## v0.2.0の変更

- Qwen3のファイル／フォルダボタンを同じ寸法で横並びにし、間隔を統一。
- 01はモデル3項目を中心に表示。任意の指定やトークン長は折りたたみ。
- 独立したFlowタブを廃止。03の「高度な設定」へ集約。05はサンプル。
- 選択肢がLoRAだけの学習方式欄と、適用されないmode_scaleを削除。旧設定は読込時に移行。
- Optimizer 14種類、Scheduler 10種類、追加引数と専用パラメータを追加。
- 新規初期値を参照TrainerのGUIテンプレートに変更。既存プロジェクトは維持。

## 対応範囲

- 28ブロックのAnimaに対するDiT LoRA。Qwen3 / LLM Adapter / VAEは凍結します。
- **Anima 2.9B（40層）のLoRA学習は非対応です。** モデル選択時に40層を検出した場合は学習前チェックで停止します。
- SageAttentionは学習用として非対応です。torch SDPA / xFormers / Flash Attentionを使用してください。
- LoRA重み・optimizer state保存、state再開、総学習上限の延長、LoRA重みを初期値にした新規学習。
- 日本語／英語、プロジェクト保存・コピー・インポート、設定プレビューとエクスポート。
- アスペクト比bucket、画像／caption確認、キャッシュ、CPU block swap、学習中サンプル、進捗／ログ／VRAM表示。
- Kohaku、LoKr、LoHa、量子化DiT、MOSS専用オプションは含みません。

## 初期値

新規プロジェクトの初期値は以下です。

Rank 16 / Alpha 16、AdamW8bit、基本LR 0.00005、weight decay 0.01、cosine、warmup 0、1024px、batch 1、2000 STEP、100 STEPごとに保存、caption dropout 0.05、workers 4、seed 42。各キャッシュON、gradient checkpointing ON、swap 0、compile OFF。

既存のDiT専用プロジェクトの保存値は維持します。初期値の出典と変更履歴は[初期値の出典](docs/DEFAULTS_SOURCE.md)に記載しています。

初期値は開始用の案です。特定GPUでのVRAM容量、学習速度、品質を保証するものではありません。VRAMに合わせて解像度やswapを手動で調整できます。高速化ボタンは解像度・Rank・学習率・学習期間を変更しません。

## 保存先

- `data/`: プロジェクト、表示設定、実行環境の指定。
- `data/projects/<ID>/runs/<日時_ID>/models/`: 学習したLoRAとstate。
- 各runの`training.log`、`train.toml`、`dataset.toml`、設定・環境snapshotで実行条件を確認できます。
- `data/projects/<ID>/cache/`: GUI専用キャッシュ。元画像フォルダの既存npzを再利用・削除しません。モデル、トークン長、caption、画像や設定の変更でキャッシュ先を変更します。不要な旧キャッシュは学習停止中に削除できます。
- `backend/anima-env/`: GUI専用の学習環境。MOSS版の環境と分離しています。

`ANIMA_TRAINER_DATA`環境変数でdataの保存先を変更できます。更新時は`data/`、`backend/`、モデルと画像を保持してください。MOSSのプロジェクトはモデル構成が異なるため読み込めません。

## 学習エンジン

sd-scriptsは検証対象のcommit `cb70b96858477e716a2096292ab8aab433277f38`に固定します。インストーラーはCUDA版PyTorch、requirements、TensorBoard等を専用venvに導入し、CUDA / bf16とAnimaスクリプトを検査してから環境を登録します。

標準AttentionはPyTorch。Tritonは通常学習の必須条件ではありません。torch.compileは詳細設定で既定OFFです。使用時は対応するTriton（Windowsではtriton-windows）とビルド環境を専用venvへ別途用意してください。xFormers／Flash Attentionは任意導入です。

## 学習条件を変えない高速化

「04 VRAM・速度」の「学習条件を保って速度設定」はLR・Rank・Alpha・解像度・batch・STEP数・保存間隔を変更せず、キャッシュ、画像専用2D VAE、persistent workers、cuDNNカーネル選択、H2D-only swap、Fused AdamWを有効化します。後ろ2項目はswap使用時／AdamW選択時だけ適用されます。既定のAdamW8bitは変更しません。

caption shuffleや画像拡張が有効な場合はキャッシュで学習内容が変わるため、一括適用せず案内します。compileやAttentionも勝手に切り替えません。

長時間学習にはtorch.compileのdefault / max-autotune等を比較できます。「かんたん導入」のTritonチェックで対応版を導入しGPUテストできます。初回コンパイル時間・追加VRAM・WindowsのC++ビルド環境が必要になる場合があります。TF32は数値精度を変えるため既定OFFです。

これらは学習ハイパーパラメータを保ちますが、カーネルや演算順序による浮動小数点の丸め差は発生し得ます。速度改善率は実モデルで未計測です。

再開カウンター・state保存と任意の別LLM Adapter読込は、起動時にメモリ上の互換処理を適用します。インストール済みsd-scripts自体は変更しません。対応箇所が変わった場合はエラーで停止します。

## Optimizer / Scheduler

Optimizer: AdamW / AdamW8bit / PagedAdamW8bit / PagedAdamW32bit / Lion / Lion8bit / PagedLion8bit / SGDNesterov / CAME / Adafactor / Prodigy / DAdaptAdam / AdamWScheduleFree / RAdamScheduleFree。

Scheduler: constant / constant_with_warmup / linear / cosine / cosine_with_restarts / cosine_with_min_lr / polynomial / inverse_sqrt / warmup_stable_decay / adafactor。

03の「高度な設定」でWarmup、最小学習率の比率、減衰STEP数、Timescale、追加引数を設定できます。追加引数は1行1つの`key=値`。例: `betas=(0.9, 0.99)`、`eps=1e-8`。専用入力欄のある値の重複指定は拒否します。

ScheduleFreeはconstant / Warmup 0、adafactor SchedulerはAdafactorとの組み合わせが必要です。選択時にも入力済みのLRは変更しません。Prodigy / DAdaptAdamのLRはAdamWとは意味が異なるため、表示される説明を確認して調整してください。

追加Optimizerのパッケージが既存の専用環境にない場合は「かんたん導入」で同じ環境を確認・修復してください。学習前に必要パッケージをチェックします。

## 再開

「03 学習」→「stateから再開・追加学習」で選択します。元のモデル・データ・Rank・Optimizer等を復元し、必要なら総STEP上限を変更します。上限変更時は新しい総上限でLRを再計算するため、再開時のLRが変わる場合があります。Kohakuのcosine延長処理とは異なります。

ScheduleFree / adafactor Schedulerではstateの総上限変更は非対応です。同じ計画で再開するか、LoRA重みから新規の追加学習を行ってください。

停止ボタンはプロセスを停止します。停止時のstate保存は保証されません。最後に正常保存されたstateを使用してください。「stateは最新の1つだけ保持」は今回の実行で正常保存した古いstateだけを削除し、LoRAや再開元は削除しません。

## Animaの学習対象

`network_train_unet_only=True`で起動します。sd-scripts共通の名前では「U-Netのみ」ですが、Animaの場合はDiT上のLoRAを指します。DiT本体の元の重み、Qwen3 Text Encoder、LLM Adapter、VAEは更新しません。Qwen3の学習はTEキャッシュを無効にしても有効になりません。

「01 モデル」の別LLM Adapterファイル指定は凍結した重みの読み込み用です。学習を有効にする設定ではありません。旧版でLLM Adapter学習を有効にしたプロジェクト／stateの継続は、この版では拒否します。新しいDiT専用プロジェクトを作成してください。

## 層別LR・正則化

「03 学習」→「高度な設定」で設定します。基本LRは0.00005。Self-Attention / Cross-Attention / MLPは空欄で基本LRを使い、正数で上書き、0で該当系統のLoRAを作成しません。詳細な正規表現LRは正数の層別LRより優先しますが、0指定の凍結は解除しません。

「キャラ層別LRの例」などの適用ボタンは層別LRだけを変更します。Rank、Alpha、基本LR、batch、解像度、STEP数は変更しません。記事の例を比較用に用意しており、最適値を保証するプリセットではありません。詳細は[Anima設定ガイド](docs/ANIMA_SETTINGS.md)を参照してください。

「02 データセット」でサブセットを正則化画像として指定できます。最低1つの教師画像サブセットが必要です。正則化画像のLoss重みは既定1。正則化画像は別キャラ／背景などを含め、目的に合うものを手動で用意してください。データセットを自動生成したりcaptionを書き換えたりはしません。

Neuron / Rank / Module dropoutとLoRA weight norm上限は既定0で無効です。相対STEP方式のAdafactorとScheduleFreeでは正数の層別LRを拒否し、設定を無視したまま起動しないようにしています。

## 出力の利用

DiTのみの標準Anima LoRAを生成環境で確認してください。Qwen3用LoRAの変換機能、MOSS用QKV変換、EMA変換は搭載していません。

## 検証

`python -m unittest discover -s tests -v`。検証記録は`VALIDATION.md`を参照。モデル本体は同梱しません。実モデルを使った学習完走・生成品質検証は別途必要です。

## 出典

- GUI原作: [suigin10/MOSS-LoRA-Trainer](https://github.com/suigin10/MOSS-LoRA-Trainer) (MIT)
- 学習エンジン: [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)
- [Animaモデル](https://huggingface.co/circlestone-labs/Anima)
- [Anima学習ガイド](https://github.com/kohya-ss/sd-scripts/blob/main/docs/anima_train_network.md)

独立した派生GUIです。モデルと学習エンジンは各配布元のライセンスに従って利用してください。
