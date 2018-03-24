import cv2,datetime,sys,glob
import numpy as np
import  matplotlib.pyplot as plt
import matplotlib.cm as cm

from keras.models import Sequential,model_from_json#json格式存放
from keras.layers import Dense,Dropout,Activation,merge,Flatten
from keras.callbacks import EarlyStopping #在合适的位置尽早结束，梯度下降的技巧
from keras.layers.convolutional import Conv2D,Conv3D

#计算PSNR
def psnt(A,B):
    return 10*np.log(255*255.0/(((A.astype(np.float)-B)**2).mean()))/np.log(10)

def double2uint8(I,ratio=1.0):
    return np.clip(np.round(I*ratio),0,255).astype(np.uint8)

def GetNlmData(I,templateWindowSize=4,searchWindow=9):
    f=int(templateWindowSize/2)
    t=int(searchWindow/2)
    height,width=I.shape[:2]
    padLength=t+f
    I2=np.pad(I,padLength,'symmetric')#滑动时边界，将边缘对称折叠上去
    I_=I2[padLength-f:padLength+f+height,padLength-f:padLength+f+width] #注意边界

    res=np.zeros((height,width,templateWindowSize+2,t+t+1,t+t+1))#有问题？%这段主要是控制不超出索引值
    # 其实主要是将各种参数放到一个矩阵中，便于计算
    for i in range(-t,t+1):#大的滑动窗
        for j in range(-t,t+1):
            I2_=I2[padLength+i-f:padLength+i+f+height,padLength+j-f:padLength+f+j+width]#某个图像块
            for kk in range(templateWindowSize):#计算得到一个高斯核,分布权重
                kernel=np.ones((2*kk+1,2*kk+1))
                kernel=kernel/kernel.sum()#进行归一化
                res[:, :, kk, i+t, j+t] = cv2.filter2D((I2_-I_)**2,-1,kernel)[f:f+height,f+width]
            res[:,:,-2,i+t,j+t]=I2_[f:f+height,f:f+width]-I
            res[:,:,-1,i+t,j+t]=np.exp(-np.sqrt(i**2+j**2))
    print(res.max(),res.min())
    return res

#构建模型进行训练
def zmTrain(trainX,trainY):
    model=Sequential()
    if 1:

        model.add(Dense(200,init='uniform',input_dim=trainX.shape[1]))
        model.add(Activation('relu'))
        model.add(Dense(100))
        model.add(Activation('relu'))
        model.add(Dense(1))
        model.summary()
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    else:
        with open('model.json','rb') as fd:
            model=model_from_json(fd.read())
            model.load_weights('weight.h5')
            model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
    early_stopping=EarlyStopping(monitor='val_loss',patience=5)
    hist=model.fit(trainX,trainY,batch_size=150,epochs=200,shuffle=True,verbose=1,
                   validation_split=0.1,callbacks=[early_stopping])
    print(hist.history)

    res=model.predict(trainX)
    res=np.clip(np.round(res.ravel()*255),0,255)#将res拉直，并进行归一化
    print(psnt(res,trainY*255))
    return model
if __name__=='__main__':
    sigma=20.0
    if 1:
        trainX=None
        trainY=None

        for d in glob.glob('*.jpg'):#路径查找
            I=cv2.imread(d,0)
            I1=double2uint8(I+np.random.randn(*I.shape) *sigma)
            data=GetNlmData(I1.astype(np.double)/255)
            s=data.shape
            data.resize((np.prod(s[:2]),np.prod(s[2:])))

            if trainX is None:
                trainX=data
                trainY=((I.astype(np.double)-I1)/255).ravel()
            else:
                trainX=np.concatenate((trainX,data),axis=0)
                trainY=np.concatenate((trainY,((I.astype(np.double)-I1)/255).ravel()),axis=0)

        model=zmTrain(trainX,trainY)
        with open('model.json','wb')as fd:
            fd.write(bytes(model.to_json(),'utf8'))
        model.save_weights('weight.h5')
    if 1:#滤波
        with open('model.json','rb') as fd:
            model=model_from_json(fd.read().decode())
            model.load_weights('weight.h5')
        I=cv2.imread('F:/noise_image/lena.jpg',0)
        I1=double2uint8(I+np.random.randn(*I.shape)*sigma)

        data=GetNlmData(I1.astype(np.double)/255)
        s=data.shape
        data.resize((np.prod(s[:2]) ,np.prod(s[2:])))
        res=model.predict(data)
        res.resize(I.shape)
        res=np.clip(np.round(res*255+I1),0,255)
        print('nwNLM PSNR',psnt(res,I))
        res=res.astype(np.uint8)
        cv2.imwrite('F:/noise_image/Outlena10.bmp',res)









