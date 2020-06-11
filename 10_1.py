import tensorflow as tf
import numpy as np
x=np.array([137.97,104.50,100.00,124.32,79.20,99.00,124.00,114.00,106.69,138.05,53.75,46.91,68.00,63.02,81.26,86.21])
y=np.array([145.00,110.00,93.00,116.00,65.32,104.00,118.00,91.00,62.00,133.00,51.00,45.00,78.50,69.65,75.69,95.30])

# 设置超参数
learn_rate=0.0001
iter=10
display_step=1

np.random.seed(612)
w=tf.Variable(np.random.randn())
b=tf.Variable(np.random.randn())

mse=[]
for i in range(0,iter+1):
    with tf.GradientTape() as tape:
        pred=w*x+b
        Loss=0.5*tf.reduce_mean(tf.square(y-pred))
    mse.append(Loss)
    dL_dW,dL_db=tape.gradient(Loss,[w,b])
    w.assign_sub(learn_rate*dL_dW)
    b.assign_sub(learn_rate*dL_db)
    if i % display_step==0:
        print("i:%i, Loss:%f ,w:%f ,b:%f" % (i,Loss,w.numpy(),b.numpy()))