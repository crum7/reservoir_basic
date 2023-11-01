# Reservoir Computing with Untrained Convolutional Neural Networks for Image Recognitionの実装

# 概要
- CNNの畳み込み層とプーリング層をESNの前処理として使用する。
- 具体的には、CNNで画像の特徴を抽出し、全結合層の部分をリザバーコンピューティングで行って、ESNで画像認識を行っている。
- CNN部分では、学習を行わない。