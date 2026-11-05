# Anima設定ガイド

## 学習範囲

DiTのLoRAのみを学習します。Qwen3 Text Encoder、LLM Adapter、VAE、元のDiT重みは凍結します。共通CLIの`network_train_unet_only`はAnimaではこの意味です。TEキャッシュのON/OFFでは学習範囲は変わりません。

## 層別LR

03の高度な設定からSelf-Attention、Cross-Attention、MLPを指定できます。空欄は基本LR、0はその系統を無効にします。0の場合はLoRAを作らず計算も省きます。詳細なregex LRは正数の層別LRより優先し、凍結は解除しません。include / excludeは従来通りモジュールの選択に適用されます。

比較ボタンは層別LRだけを変更します。他の学習条件は保持します。最適値を断定するものではありません。

## 正則化

02で正則化画像用サブセットにチェックを入れます。Loss重みはそのサブセットだけに適用します。教師画像サブセットを少なくとも1つ用意してください。正則化画像の不足分の補充やbucket端数で、単純な画像枚数計算と実際のSTEP数が異なる場合があります。

Neuron / Rank / Module dropout、LoRA weight norm上限も使用できます。既定0で無効です。これらはsd-scriptsの標準機能で、記事の未公開部分の実験結果を再現したものではありません。

## 参考

[賢木イオ「Anima用LoRA学習を大検証 キャラLoRAの層別学習から正則化まで」](https://note.com/studiomasakaki/n/nf39775327336)の公開部分を参照し、層別LRの操作と比較用設定を追加しました。記事の全設定をデフォルトへ上書きせず、選択した設定だけを適用します。後半の正則化検証は公開範囲外のため、その結果は推測していません。

実装は採用した[sd-scriptsのAnima LoRAモジュール](https://github.com/kohya-ss/sd-scripts/blob/cb70b96858477e716a2096292ab8aab433277f38/networks/lora_anima.py)と[データセット設定](https://github.com/kohya-ss/sd-scripts/blob/cb70b96858477e716a2096292ab8aab433277f38/library/config_util.py)を確認しています。フルファインチューニング用のself_attn_lr引数ではなく、LoRA用のnetwork_reg_lrs / network_reg_dimsへ変換します。

Anima 2.9B（40層）とSageAttention学習は非対応です。
