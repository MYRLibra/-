'''
感知机算法的原始形式
eta_为步长/学习率,一般设为1
'''
import numpy as np

class perceptron:
    def __init__(self,eta_=1):
        self.eta_=eta_        

    def fit(self,x_data,y_data):
        global w,b
        flag=True #训练集中存在误分类点
        w=[0]*x_data.shape[1]
        b=0
        while(flag):
            for i in range(x_data.shape[0]):
                if self.cal(x_data[i],y_data[i])<=0:
                    print(w,b)
                    self.update(x_data[i],y_data[i])
                    break
                elif i+1==x_data.shape[0]:
                    print(w,b)
                    flag=False #训练集中不存在误分类点

    def cal(self,x,y):
        global w,b
        res=0
        for i in range(len(x)):
            res+=x[i]*w[i]
        res+=b
        res*=y
        return res
    
    def update(self,x,y):
        global w,b
        for i in range(len(x)):
            w[i]+=self.eta_*y*x[i]
        b+=self.eta_*y


if __name__=="__main__":
    x_data=np.array([[3,3],[4,3],[1,1]])
    y_data=[1,1,-1]
    clf=perceptron(1)
    clf.fit(x_data,y_data)

