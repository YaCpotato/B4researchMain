# deepaugment
## ソース構成
* augmenter
* build_features
* childcnn
* controller
* deepaugment
* image_generator
* notebook
* objective
* run_full_model
* run
* wide_resnet

## augmenter
#### normalize(data)
データ(data)を255で割る
```
    Normalization(正規化)-----
    データの尺度を統一すること
    最低０、最高１になるようにデータを加工すること
    -------------------------
```

#### denormalize()
データに255を掛ける

#### transform(aug_type, magnitude, X)
加工の種類(aug_type,20種類のうちひとつ)と、重み(magnitude,0~1)、データの一部を受け取り、加工をする
#### augment_by_policy(X, y, *hyperparams)
未使用


## build_features
### DataOpクラス
#### load(dataset_name)
keras.datasetsにdataset_nameできた文字列に一致するattributeがあったら、
データをロードし、train dataとvalidation data,train labelとvalidation       labelを結合し、入力shapeとともに渡す
    
#### preprocess_normal(data)
データXを正規化(normalize)し、ラベルYをワンホットエンコーディングにして返す

#### split_train_val_sets(X, y, train_set_size, val_set_size)
train_set_size分のデータをXから取り出し、yから対応するラベルデータを取り出し、返す

#### preprocess(X, y, train_set_size, val_set_size=1000)
split_train_val_setsで取り出されたデータに対しデータの正規化とワンホットエンコーディングを行い、返す
#### sample_validation_set(data)
一定枚数、評価に使うデータとラベルを取り出す(毎回同じ評価用データでepochを回さない)
#### find_num_classes(data)
クラス数を返す

## childcnn
### ChildCNNクラス
#### __init__(self, input_shape=None, num_classes=None, config=None)
#### fit(self, data, augmented_data=None, epochs=None)
#### fit_normal(self, data, epochs=None, csv_logger=None)
#### fit_with_generator(self, datagen, X_val, y_val, train_data_size, epochs=None, csv_logger=None)
#### load_pre_augment_weights(self)
#### evaluate_with_refreshed_validation_set(self, data)
#### create_child_cnn(self)
ここからbuild_wrnやbuild_prepared_modelの呼び出しを行える
#### build_prepared_model(self)
mobilenetやinceptionv3のモデルを使用したい時に呼ぶ
#### build_wrn(self)
WideResNetを作る
* depth(ResidualBlockいくつ繋げるか)
    * wrn_[40]_4
* width(パラメータを何倍するか)
    * wrn_40_[4]
* dropout_rate
    * 0.0(dropout層なし？)
* include_top
    * 
* weights
    * 重み
* input_tensor
    * None
* input_shape(入力シェイプ)
    * self.input_shape
* classes(クラス数)
    * self.num_classes
* activation(活性化関数)
    * softmax
* Optimizer
    * lr: 0以上の浮動小数点数．学習率
        * 0.001
    * beta_1: 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です．
        * 0.9
    * beta_2: 浮動小数点数, 0 < beta < 1. 一般的に1に近い値です．
        * 0.999
    * epsilon: 0以上の浮動小数点数．微小量．NoneならばデフォルトでK.epsilon()．
        * None
    * decay: 0以上の浮動小数点数．各更新の学習率減衰．
        * 0.0
    * amsgrad: 論文"On the Convergence of Adam and Beyond"にあるAdamの変種であるAMSGradを適用するかどうか．
        * False
    * clipnorm documentationなし
        * 1.0

#### build_basicCNN(self)
俗に名前がついているわけでもない普通のCNNを作る。optimizerはRMSProp,誤差関数は交差エントロピー

## augmenter
加工のタイプは、
|deepaugment|Autoaugment|Effect|
|:--:|:--:|:--:|
|crop|-|拡大|
|gaussian-blur|-|ノイズ|
| rotate|rotate|回転|
| shear|ShearX,ShearY|斜めにする|
| translate-x|-|?|
| translate-y|-|?|
|-|TranslateX|X軸平行移動|
|-|TranslateY|Y軸平行移動|
| horizontal-flip|-|左右反転|
| vertical-flip|-|上下反転|
| sharpen|sharpness|鮮明化|
| emboss|-|エンボスフィルタ|
| additive-gaussian-noise|-|可算性ガウシアンノイズ|
| dropout|-|?|
| coarse-dropout|-|?|
| gamma-contrast|-|ガンマ補正|
|-|Contrast|コントラスト補正|
| brighten|brightness|明るさ補正|
| invert|invert|画素値逆転|
| fog|-|霧がけ|
| clouds|-|雲ノイズ|
| histgram-equalize|Equalize|ヒストグラム平坦化|
| super-pixels|-|Superpixel|
| perspective-transform|-|透視変換|
| elastic-transform|-|弾性変換|
| add-to-hue-and-saturation|-|色相、彩度増加|
| coarse-salt-pepper|-|粗塩ノイズ|
| grayscale|-|グレースケール変換|
|-|Color(Binarization)|二値化
|-|Cutout|Cutout|
|-|AutoContrast|自動コントラスト補正|
|-|Posterize|ポスター化|
|-|Solarize|?|
|-|SamplePairng|画像の重ねがけ|
