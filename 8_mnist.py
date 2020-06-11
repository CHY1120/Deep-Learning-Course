{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "epoch=  1,train_loss=3.0829,train_acc=0.7879,val_loss=2.9345,val_acc=0.7977\n",
      "epoch=  2,train_loss=2.5351,train_acc=0.8253,val_loss=2.4335,val_acc=0.8324\n",
      "epoch=  3,train_loss=0.6326,train_acc=0.9419,val_loss=0.6533,val_acc=0.9396\n",
      "epoch=  4,train_loss=0.4576,train_acc=0.9535,val_loss=0.5349,val_acc=0.9473\n",
      "epoch=  5,train_loss=0.4081,train_acc=0.9508,val_loss=0.5037,val_acc=0.9442\n",
      "epoch=  6,train_loss=0.3089,train_acc=0.9543,val_loss=0.4004,val_acc=0.9453\n",
      "epoch=  7,train_loss=0.2642,train_acc=0.9613,val_loss=0.3986,val_acc=0.9480\n",
      "epoch=  8,train_loss=0.2103,train_acc=0.9636,val_loss=0.3218,val_acc=0.9506\n",
      "epoch=  9,train_loss=0.2025,train_acc=0.9638,val_loss=0.3286,val_acc=0.9481\n",
      "epoch= 10,train_loss=0.1783,train_acc=0.9641,val_loss=0.3205,val_acc=0.9511\n",
      "epoch= 11,train_loss=0.1688,train_acc=0.9653,val_loss=0.2979,val_acc=0.9500\n",
      "epoch= 12,train_loss=0.1300,train_acc=0.9710,val_loss=0.2620,val_acc=0.9567\n",
      "epoch= 13,train_loss=0.1166,train_acc=0.9732,val_loss=0.2712,val_acc=0.9537\n",
      "epoch= 14,train_loss=0.1280,train_acc=0.9704,val_loss=0.2842,val_acc=0.9535\n",
      "epoch= 15,train_loss=0.1214,train_acc=0.9742,val_loss=0.2884,val_acc=0.9568\n",
      "epoch= 16,train_loss=0.1220,train_acc=0.9729,val_loss=0.2936,val_acc=0.9553\n",
      "epoch= 17,train_loss=0.1160,train_acc=0.9744,val_loss=0.2988,val_acc=0.9541\n",
      "epoch= 18,train_loss=0.0920,train_acc=0.9803,val_loss=0.2682,val_acc=0.9590\n",
      "epoch= 19,train_loss=0.1114,train_acc=0.9775,val_loss=0.2902,val_acc=0.9582\n",
      "epoch= 20,train_loss=0.1057,train_acc=0.9787,val_loss=0.2924,val_acc=0.9609\n",
      "epoch= 21,train_loss=0.1015,train_acc=0.9794,val_loss=0.2960,val_acc=0.9591\n",
      "epoch= 22,train_loss=0.1208,train_acc=0.9771,val_loss=0.3208,val_acc=0.9576\n",
      "epoch= 23,train_loss=0.0843,train_acc=0.9835,val_loss=0.2824,val_acc=0.9606\n",
      "epoch= 24,train_loss=0.0807,train_acc=0.9850,val_loss=0.3060,val_acc=0.9607\n",
      "epoch= 25,train_loss=0.1089,train_acc=0.9805,val_loss=0.3445,val_acc=0.9578\n",
      "epoch= 26,train_loss=0.1046,train_acc=0.9814,val_loss=0.3467,val_acc=0.9583\n",
      "epoch= 27,train_loss=0.1406,train_acc=0.9760,val_loss=0.3857,val_acc=0.9551\n",
      "epoch= 28,train_loss=0.0993,train_acc=0.9826,val_loss=0.3419,val_acc=0.9590\n",
      "epoch= 29,train_loss=0.0882,train_acc=0.9839,val_loss=0.3591,val_acc=0.9594\n",
      "epoch= 30,train_loss=0.0935,train_acc=0.9841,val_loss=0.3423,val_acc=0.9611\n",
      "epoch= 31,train_loss=0.0860,train_acc=0.9839,val_loss=0.3361,val_acc=0.9622\n",
      "epoch= 32,train_loss=0.0800,train_acc=0.9856,val_loss=0.3411,val_acc=0.9605\n",
      "epoch= 33,train_loss=0.0722,train_acc=0.9876,val_loss=0.3675,val_acc=0.9616\n",
      "epoch= 34,train_loss=0.0968,train_acc=0.9837,val_loss=0.3826,val_acc=0.9601\n",
      "epoch= 35,train_loss=0.1019,train_acc=0.9850,val_loss=0.3865,val_acc=0.9604\n",
      "epoch= 36,train_loss=0.1134,train_acc=0.9831,val_loss=0.4041,val_acc=0.9586\n",
      "epoch= 37,train_loss=0.0711,train_acc=0.9875,val_loss=0.3428,val_acc=0.9637\n",
      "epoch= 38,train_loss=0.0916,train_acc=0.9851,val_loss=0.4044,val_acc=0.9588\n",
      "epoch= 39,train_loss=0.1251,train_acc=0.9812,val_loss=0.4075,val_acc=0.9593\n",
      "epoch= 40,train_loss=0.1035,train_acc=0.9849,val_loss=0.4078,val_acc=0.9597\n",
      "epoch= 41,train_loss=0.0930,train_acc=0.9852,val_loss=0.3956,val_acc=0.9609\n",
      "epoch= 42,train_loss=0.0889,train_acc=0.9867,val_loss=0.4097,val_acc=0.9619\n",
      "epoch= 43,train_loss=0.0823,train_acc=0.9876,val_loss=0.3903,val_acc=0.9611\n",
      "epoch= 44,train_loss=0.0654,train_acc=0.9907,val_loss=0.3699,val_acc=0.9641\n",
      "epoch= 45,train_loss=0.1058,train_acc=0.9859,val_loss=0.4238,val_acc=0.9601\n",
      "epoch= 46,train_loss=0.0824,train_acc=0.9882,val_loss=0.4087,val_acc=0.9628\n",
      "epoch= 47,train_loss=0.0818,train_acc=0.9881,val_loss=0.4142,val_acc=0.9610\n",
      "epoch= 48,train_loss=0.0733,train_acc=0.9904,val_loss=0.4178,val_acc=0.9627\n",
      "epoch= 49,train_loss=0.1207,train_acc=0.9842,val_loss=0.4455,val_acc=0.9593\n",
      "epoch= 50,train_loss=0.0926,train_acc=0.9879,val_loss=0.4118,val_acc=0.9624\n",
      "epoch= 51,train_loss=0.0878,train_acc=0.9879,val_loss=0.4532,val_acc=0.9590\n",
      "epoch= 52,train_loss=0.0817,train_acc=0.9891,val_loss=0.4144,val_acc=0.9625\n",
      "epoch= 53,train_loss=0.0623,train_acc=0.9915,val_loss=0.3882,val_acc=0.9656\n",
      "epoch= 54,train_loss=0.0561,train_acc=0.9921,val_loss=0.3943,val_acc=0.9643\n",
      "epoch= 55,train_loss=0.0833,train_acc=0.9891,val_loss=0.4288,val_acc=0.9641\n",
      "epoch= 56,train_loss=0.1007,train_acc=0.9878,val_loss=0.4645,val_acc=0.9608\n",
      "epoch= 57,train_loss=0.0918,train_acc=0.9883,val_loss=0.4193,val_acc=0.9637\n",
      "epoch= 58,train_loss=0.1181,train_acc=0.9864,val_loss=0.4635,val_acc=0.9602\n",
      "epoch= 59,train_loss=0.0552,train_acc=0.9930,val_loss=0.4316,val_acc=0.9639\n",
      "epoch= 60,train_loss=0.0674,train_acc=0.9911,val_loss=0.4355,val_acc=0.9621\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "# %matplotlib inline\n",
    "\n",
    "mnist = tf.keras.datasets.mnist\n",
    "(train_images,train_labels),(test_images,test_labels) = mnist.load_data()\n",
    "total_num = len(train_images)\n",
    "valid_split = 0.2\n",
    "train_num = int(total_num*(1-valid_split))\n",
    "\n",
    "train_x = train_images[:train_num]\n",
    "train_y = train_labels[:train_num]\n",
    "\n",
    "valid_x = train_images[train_num:]\n",
    "valid_y = train_labels[train_num:]\n",
    "\n",
    "test_x = test_images\n",
    "test_y = test_labels\n",
    "\n",
    "train_x = train_x.reshape(-1,784)\n",
    "valid_x = valid_x.reshape(-1,784)\n",
    "test_x = test_x.reshape(-1,784)\n",
    "\n",
    "train_x = tf.cast(train_x/255.0,tf.float32)\n",
    "valid_x = tf.cast(valid_x/255.0,tf.float32)\n",
    "test_x = tf.cast(test_x/255.0,tf.float32)\n",
    "\n",
    "train_y = tf.one_hot(train_y,depth=10)\n",
    "valid_y = tf.one_hot(valid_y,depth=10)\n",
    "test_y = tf.one_hot(test_y,depth=10)\n",
    "\n",
    "Input_Dim = 784\n",
    "H1_NN = 64\n",
    "W1 = tf.Variable(tf.random.normal([Input_Dim,H1_NN],mean=0.0,stddev=1.0,dtype=tf.float32))\n",
    "B1 = tf.Variable(tf.zeros([H1_NN]),dtype=tf.float32)\n",
    "\n",
    "Output_Dim = 10\n",
    "W2 = tf.Variable(tf.random.normal([H1_NN,Output_Dim],mean=0.0,stddev=1.0,dtype=tf.float32))\n",
    "B2 = tf.Variable(tf.zeros([Output_Dim]),dtype=tf.float32)\n",
    "\n",
    "W = [W1,W2]\n",
    "B = [B1,B2]\n",
    "\n",
    "def model(x,w,b):\n",
    "    x = tf.matmul(x,w[0])+b[0]\n",
    "    x = tf.nn.relu(x)\n",
    "    x = tf.matmul(x,w[1])+b[1]\n",
    "    pred = tf.nn.softmax(x)\n",
    "    return pred\n",
    "def loss(x,y,w,b):\n",
    "    pred = model(x,w,b)\n",
    "    loss_ = tf.keras.losses.categorical_crossentropy(y_true=y,y_pred=pred)\n",
    "    return tf.reduce_mean(loss_)\n",
    "training_epochs = 60 #训练轮数\n",
    "batch_size = 50 #单次训练样本数\n",
    "learning_rate = 0.01 #学习率\n",
    "\n",
    "def grad(x,y,w,b):\n",
    "    with tf.GradientTape() as tape:\n",
    "        loss_ = loss(x,y,w,b)\n",
    "    return tape.gradient(loss_,w+b)\n",
    "optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)\n",
    "def accuracy(x,y,w,b):\n",
    "    pred = model(x,w,b)\n",
    "    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))\n",
    "    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))\n",
    "\n",
    "steps = int(train_num/batch_size)\n",
    "loss_list_train = []\n",
    "loss_list_valid = []\n",
    "acc_list_train = []\n",
    "acc_list_valid = []\n",
    "\n",
    "for epoch in range(training_epochs):\n",
    "    for step in range(steps):\n",
    "        xs = train_x[step*batch_size:(step+1)*batch_size]\n",
    "        ys = train_y[step*batch_size:(step+1)*batch_size]\n",
    "        grads = grad(xs,ys,W,B)\n",
    "        optimizer.apply_gradients(zip(grads,W+B))\n",
    "\n",
    "    loss_train = loss(train_x,train_y,W,B).numpy()\n",
    "    loss_valid = loss(valid_x,valid_y,W,B).numpy()\n",
    "    acc_train = accuracy(train_x,train_y,W,B).numpy()\n",
    "    acc_valid = accuracy(valid_x,valid_y,W,B).numpy()\n",
    "    loss_list_train.append(loss_train)\n",
    "    loss_list_valid.append(loss_valid)\n",
    "    acc_list_train.append(acc_train)\n",
    "    acc_list_valid.append(acc_valid)\n",
    "    print(\"epoch={:3d},train_loss={:.4f},train_acc={:.4f},val_loss={:.4f},val_acc={:.4f}\".format(epoch+1,loss_train,acc_train,loss_valid,acc_valid))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
