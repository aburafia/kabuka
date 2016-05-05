# -*- coding: utf-8 -*-

import tensorflow as tf
import kabuka_input
import time

# 開始時刻
start_time = time.time()
print "開始時刻: " + str(start_time)

#株価データの読み込み
#70%の訓練データと、20%のテストデータ、10%の判定用データを作成している
# kabuka.train.kabukasは[件数,14*5]の配列であり、kabuka.train.lablesは[件数, 2]の配列
# lablesの配列は、対応する株価のデータの２日前の始め値よりも、前日の終値が高ければ、[1,0]となっている
print "--- 株価データの読み込み開始 ---"
kabuka = kabuka_input.read_data_sets()
print "--- 株価データの読み込み完了 ---"

# 訓練株価を入れる変数 14日分の、始、終、高、安、出来高
x = tf.placeholder(tf.float32, [None, 14 * 5])

# 重み
# 訓練データの次元、ラベル（0-1の２値）数の列の行列
# 初期値として0を入れておく
W = tf.Variable(tf.zeros([14*5, 2]))

# バイアス
# ラベル数の列の行列
# 初期値として0を入れておく
b = tf.Variable(tf.zeros([2]))

# シグも井戸回帰を実行
# matmul関数で行列xとWの掛け算を行った後、bを加算する。
# yは[1, 10]の行列
y = tf.nn.sigmoid(tf.matmul(x, W) + b)

# 交差エントロピー
# y_は正解データのラベル
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

# 勾配硬化法を用い交差エントロピーが最小となるようyを最適化する
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 用意した変数Veriableの初期化を実行する
init = tf.initialize_all_variables()

# Sessionを開始する
# runすることで初めて実行開始される（run(init)しないとinitが実行されない）

sess = tf.Session()
sess.run(init)

# 1000回の訓練（train_step）を実行する
# next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
# 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
# 100つでも同じような結果を得ることができる
# feed_dictでplaceholderに値を入力することができる
print "--- 訓練開始 ---"
for i in range(1000):
    batch_xs, batch_ys = kabuka.train.next_batch(100)
    
    #print(batch_xs.shape)
    #print(batch_ys.shape)    
    
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
print "--- 訓練終了 ---"

# 正しいかの予測
# 計算された画像がどの数字であるかの予測yと正解ラベルy_を比較する
# 同じ値であればTrueが返される
# argmaxは配列の中で一番値の大きい箇所のindexが返される
# 一番値が大きいindexということは、それがその数字である確率が一番大きいということ
# Trueが返ってくるということは訓練した結果と回答が同じということ
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# 精度の計算
# correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
# Trueならば1、Falseならば0に変換される
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 精度の実行と表示
# テストデータの画像とラベルで精度を確認する
# ソフトマックス回帰によってWとbの値が計算されているので、xを入力することでyが計算できる
print "精度"
print(sess.run(accuracy, feed_dict={x: kabuka.test.kabukas, y_: kabuka.test.labels}))

# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)
print "かかった時間: " + str(end_time - start_time)
