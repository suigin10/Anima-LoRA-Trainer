# 新規初期値の出典

参照: [gazingstars123/Anima-Standalone-Trainer](https://github.com/gazingstars123/Anima-Standalone-Trainer)
確認commit: `7f5c336f3c61802e9b7a64fb4a4492d50362a90f6`

採用元は次のGUI用テンプレートです。ルートにあるAnima_lora_configs.tomlとは値が異なります。
- [config_template.toml](https://github.com/gazingstars123/Anima-Standalone-Trainer/blob/7f5c336f3c61802e9b7a64fb4a4492d50362a90f6/training-ui/templates/config_template.toml)
- [dataset_template.toml](https://github.com/gazingstars123/Anima-Standalone-Trainer/blob/7f5c336f3c61802e9b7a64fb4a4492d50362a90f6/training-ui/templates/dataset_template.toml)

| 項目 | 新規初期値 |
|---|---|
| Rank / Alpha | 16 / 16 |
| Optimizer / LR | AdamW8bit / 0.00005 |
| Weight decay | 0.01 |
| Scheduler / Warmup | cosine / 0 |
| 学習期間 / 保存 | 2000 STEP / 100 STEP |
| 解像度 / batch | 1024 / 1 |
| bucket min / max / step | 512 / 1536 / 64 |
| Caption dropout | 0.05 |
| repeats / shuffle / 色拡張 | 1 / OFF / OFF |
| workers / 勾配累積 / seed | 4 / 1 / 42 |
| checkpointing / キャッシュ | ON / ON |
| LLM Adapter学習 | OFF |
| サンプル間隔 | 1 epoch |

## バックエンド間の対応

参照先の`logit_normal`は標準正規分布をsigmoid変換し、flow shift 3を適用します。本GUIの公式sd-scriptsでは同じ分布を`--timestep_sampling=shift --sigmoid_scale=1 --discrete_flow_shift=3`で指定します。参照先で`weighting_scheme=logit_normal`はlossの追加重み付けを行わないため、こちらでは実効動作が同じ`none`とします。

参照先のMuon・分散学習・独自freeze設定は、採用した公式sd-scriptsと実装が異なるためコピーしません。Qwen3は凍結するのでtext_encoder_lrも表示しません。単一LoRA方式選択と使われないmode_scaleは削除しています。

画像専用2D VAEやGPU実行側の高速化、state管理は本GUIの機能として維持しています。サンプルの内容・生成条件は参照先の空テンプレートから取得できないため本GUI側の値を保持します。

この初期値は新規作成だけに適用します。保存済みプロジェクトやstateの学習条件は変更しません。
Anima 2.9B（40層）は非対応です。初期値の採用は、そのモデルへの対応を意味しません。

## v0.3.0の上書き

ユーザー指定により、基本LRは0.00005、学習期間は2000 STEP、保存は100 STEPごとへ変更しました。50〜100枚・batch 1・勾配累積1を想定した比較開始用です。学習解像度もユーザー指定により1024pxへ変更しました。その他の共通初期値は維持しています。LLM Adapterも学習しない方針に固定しました。

層別LR・正則化のUIを追加しましたが、個別LRは既定で空欄（基本LRを使用）、追加dropout / norm正則化は既定0です。追加機能を入れただけで、保存済みの学習条件へ記事の値を自動適用することはありません。
