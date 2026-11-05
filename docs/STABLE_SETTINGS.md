# Anima 設定ガイド

01 モデル
- Anima DiT: 28ブロックの非量子化safetensors。Anima 2.9B（40層）は非対応。LLM Adapter内蔵版を推奨。
- Qwen3-0.6B: 単体safetensors、または重み・config・tokenizerを含むHugging Faceフォルダ。
- Qwen-Image VAE: Anima用の公式重みを指定。
- T5 tokenizer: 通常は空欄。T5本体の重みは不要。
- 別LLM Adapter: 通常は空欄。指定する場合はllm_adapter.*キーだけのsafetensors。

入手先: https://huggingface.co/circlestone-labs/Anima

02 データセット
- 同じフォルダに画像と同名の.txt。サブフォルダは自動再帰しません。
- repeatsは1以上。captionがない画像はclass tokensを使用できます。
- 新規初期値は1024px。解像度は手動で調整可能。
- TEキャッシュとcaption shuffleは併用不可。caption全体のdropoutは併用可能。
- Latentキャッシュとcolor augmentation / random cropは併用不可。

03 学習
- 新規初期値はRank 16 / Alpha 16 / 基本LR 0.00005、AdamW8bit、weight decay 0.01。
- cosine / warmup 0、2000 STEP / 100 STEP保存。既存のDiT専用プロジェクトの値は保持。
- save_stateは追加の容量を使います。LoRA単体からの追加学習はOptimizerを新規作成。

04 VRAM・速度
- bf16 / torch Attention / gradient checkpointing / キャッシュON。
- 2D VAEは公式の同じVAE重みを使用。
- OOM時は解像度を下げ、swapを調整。28ブロック版ではswapは0〜26。
- CPU offload / Unsloth offload / block swapは併用不可。
- compile OFFが標準。ONには対応Tritonとビルド環境が必要。
- 「学習条件を保って速度設定」は学習ハイパーパラメータを変更しません。キャッシュ・2D VAE・persistent workers・cuDNNのカーネル選択等を有効化します。
- H2D-only swapはswap使用時だけ有効。Fused AdamWはAdamW選択時だけ適用し、Optimizer自体は変更しません。
- Tritonは「かんたん導入」で任意導入。compile modeはまずdefault、長時間ならmax-autotuneも比較できます。
- TF32は精度を変えるため既定OFF。cuDNNや融合演算もビット単位の一致は保証しません。

03 高度な設定（折りたたみ）
- shift方式 / shift 3。sigmoid分布にflow shiftを適用。
- DiTのLoRAのみ。Qwen3 / LLM Adapter / VAEは凍結し、学習しません。
- モジュール名はself_attn / cross_attn / mlp / llm_adapterなど。

05 サンプル
- 通常OFF。ONにすると生成時にもVRAMと時間を使用。
- 学習対象のトリガーを含むプロンプトを指定。
- 初期案: 30 steps、CFG 3、生成shift 3、seed 42。
- 生成条件は使用するAnimaモデルに合わせて調整。

各値は品質・VRAMの保証ではありません。実際の学習と生成で調整してください。

Optimizer 14種類 / Scheduler 10種類。選択時にLRは変更しません。ScheduleFreeはconstant / Warmup 0。adafactor SchedulerはAdafactor専用。最小学習率・Timescale・減衰STEP数は該当Schedulerだけ有効。追加引数は1行1つのkey=値。

SageAttentionは学習非対応です。標準はtorch SDPA。

Animaの層別LR
- Self-Attention / Cross-Attention / MLPを個別に指定。空欄は基本LR、0は学習停止。
- 「キャラ層別LRの例」などの適用は層別LRだけを変更します。
- 正規表現による詳細LRが正数の個別LRより優先。0指定の凍結は優先。

正則化
- 02のサブセットごとに正則化画像を指定。教師画像サブセットは最低1つ必要。
- 正則化画像のLoss重みは1が既定。Neuron / Rank / Module dropout、norm上限は0で無効。
- 正則化画像は手動で用意。画像内容やタグの偏りは自分で確認してください。

U-Netのみ学習という内部フラグは、AnimaではDiTのLoRAのみを指します。Text EncoderのQwen3とLLM Adapterは学習しません。
