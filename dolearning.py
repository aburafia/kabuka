# -*- coding: utf-8 -*-

import tensorflow as tf
import kabuka_input
import time

def make_model(_x, w, b, d):

    with tf.name_scope('suisoku'):
        with tf.name_scope('hidden1') as h1_scope:
            layer_1 = tf.nn.relu(tf.add(tf.matmul(_x, w['h1']), b['b1']))
            layer_1 = tf.nn.dropout(layer_1, d) # dropout
        with tf.name_scope('hidden2') as h2_scope:
            layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w['h2']), b['b2']))
        with tf.name_scope('hidden3') as h2_scope:
            layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, w['h3']), b['b3']))
        with tf.name_scope('hidden4') as h2_scope:
            layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, w['h4']), b['b4']))
        with tf.name_scope('output') as out_scope:
            model = tf.matmul(layer_4, w['out']) + b['out']

    return model

#現状での計算結果を出力する
#シグモイドで0-1にして、0.5以上?1:0とする
def get_result(model):
    result = tf.cast(tf.greater_equal(tf.nn.sigmoid(model), tf.constant(0.5)), 'float')
    return result

#損失関数
#シグモイドで勾配計算
#どれだけ乖離してるかは、損失関数結果の平均値でチェックする
def make_loss(model, y):
    with tf.name_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(model, y)
        loss = tf.reduce_mean(cross_entropy)
        tf.scalar_summary("loss", loss)
    return loss

#勾配計算。ジャンプ幅は0.5
def make_training(loss):    
  with tf.name_scope('training'):
    training = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
  return training

#正しさを算出してみる
#このあたりを参考にした。
#http://qiita.com/sergeant-wizard/items/55256ac6d5d8d7c53a5a
def make_accuracy(yosoku, seikai):
  with tf.name_scope('accuracy'):

    # 正しいかの予測
    # 計算された画像がどの数字であるかの予測と正解ラベルを比較する
    # 同じ値であればTrueが返される
    correct_prediction = tf.equal(yosoku, seikai)

    #correct_predictionはbooleanなのでfloatにキャストし、平均値を計算する
    #Trueならば1、Falseならば0に変換される
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    
    tf.scalar_summary('accuracy', accuracy)
  return accuracy
  

def run(kabuka):

    with tf.Graph().as_default():
        #placeholder
        # 訓練株価を入れる変数 14日分の、始、終、高、安、出来高 = 70
        _x = tf.placeholder(tf.float32, [None, 70], name="placeholder_x")

        # yは正解データのラベル
        _y = tf.placeholder(tf.float32, [None, 1], name="placeholder_y")
        
        _d = tf.placeholder("float", name="placeholder_d")
        
        #各層の重み
        #平均から標準偏差の２倍以上離れたところの頻度が０となるような正規分布に従うRandomな値
        #http://www.ntrand.com/jp/truncated-normal-distribution/
        w = {
            'h1': tf.Variable(tf.truncated_normal([70, 70], stddev=0.1), name="h1_w"),
            'h2': tf.Variable(tf.truncated_normal([70, 70], stddev=0.1), name="h2_w"),
            'h3': tf.Variable(tf.truncated_normal([5, 70], stddev=0.1), name="h3_w"),
            'h4': tf.Variable(tf.truncated_normal([2, 5], stddev=0.1), name="h4_w"),
            'out': tf.Variable(tf.truncated_normal([1, 2], stddev=0.1), name="out_w")
        }
        
        #各層のバイアス
        b = {
            'b1': tf.Variable(tf.constant(0.1, shape=[70]), name="h1_b"),
            'b2': tf.Variable(tf.constant(0.1, shape=[70]), name="h2_b"),
            'b3': tf.Variable(tf.constant(0.1, shape=[5]), name="h3_b"),
            'b4': tf.Variable(tf.constant(0.1, shape=[2]), name="h4_b"),
            'out': tf.Variable(tf.constant(0.1, shape=[1]), name="out_b")
        }
        
        #全部くみたてる
        #NNと訓練一式
        model = make_model(_x, w, b, _d)
        loss = make_loss(model, _y)
        training = make_training(loss)
        
        #正しさチェック用
        result = get_result(model)
        accuracy = make_accuracy(result, _y)

        #よくわからんが、いる？
        summary_op = tf.merge_all_summaries()
    
        with tf.Session() as sess:
        
            # 初期化
            sess.run(tf.initialize_all_variables())
            
            best_loss = float("inf")
            best_accuracy = 0.0
            summary_writer = tf.train.SummaryWriter('graph/k_win_odds', graph_def=sess.graph_def)
            
            #訓練GOGO
            for train_step in range(1000):

                # 1000回の訓練（train_step）を実行する
                # next_batch(100)で100つのランダムな訓練セット（画像と対応するラベル）を選択する
                # 訓練データは60000点あるので全て使いたいところだが費用つまり時間がかかるのでランダムな100つを使う
                # 100つでも同じような結果を得ることができる
                # feed_dictでplaceholderに値を入力することができる
                batch_train_x, batch_train_y = kabuka.train.next_batch(100)                
                feed_dict_train = {_x: batch_train_x, _y: batch_train_y, _d: 1.0}
                sess.run(training, feed_dict=feed_dict_train)
                
                # なにしてるかわからん
                batch_test_x, batch_test_y = kabuka.test.next_batch(100)
                feed_dict_test = {_x: batch_test_x, _y: batch_test_y, _d: 1.0}
                test_loss = sess.run(loss, feed_dict=feed_dict_test)
                if test_loss < best_loss:
                    best_loss = test_loss
                    best_accuracy = sess.run(accuracy, feed_dict=feed_dict_test)
                
                # 途中経過を出力
                if train_step % 10 == 0:
                    summary_str = sess.run(summary_op, feed_dict=feed_dict_train)
                    summary_writer.add_summary(summary_str, train_step)
                    
                    test_logits = sess.run(logits, feed_dict=feed_dict_test)
                    test_odds = sess.run(odds, feed_dict=feed_dict_test)
                    test_y = sess.run(y, feed_dict=feed_dict_test)
                    test_loss = sess.run(loss, feed_dict=feed_dict_test)
                    
                    for test_step in range(len(test_y)):
                        print "my anser:%1.0f(%+4.2f-&gt;%3.2f)"%(test_y[test_step], test_logits[test_step], test_odds[test_step]),
                        print "accurate:%1.0f"%test_y[test_step][0]
                        print "loss:%3.2f"%test_loss
                        print "accuracy:%3.2f"%test_accuracy
                
                    print "best loss:%3.2f"%best_loss
                    print "best accuracy:%3.2f"%best_accuracy


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

print "--- 訓練開始 ---"
run(kabuka)
print "--- 訓練終了 ---"

# 終了時刻
end_time = time.time()
print "終了時刻: " + str(end_time)
print "かかった時間: " + str(end_time - start_time)
