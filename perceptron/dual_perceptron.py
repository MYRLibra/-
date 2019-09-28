'''
感知机算法的对偶形式
eta_为步长/学习率,一般设为1
'''
import numpy as np


class dualPerceptron:
    def __init__(self,eta_=1):
        self.eta_=eta_

    def fit(self,x_data,y_data):
        global a,b
        a=[0]*x_data.shape[0]
        b=0
        flag=True #训练集中存在误分类点
        Gram=self.cal_gram(x_data)
        while(flag):
            for i in range(x_data.shape[0]):
                if self.cal(Gram,y_data,i)<=0:
                   print(a,b)
                   self.update(i,y_data[i])
                   break
                elif i+1==x_data.shape[0]:
                    print(a,b)
                    self.cal_w(x_data,y_data)
                    flag=False

    def cal_gram(self,x_data):
        n=x_data.shape[0]
        Gram=np.zeros((n,n)) 
        for i in range(n):
            for j in range(n):
                ans=0
                for m in range(x_data.shape[1]):
                    ans+=x_data[i][m]*x_data[j][m]
                Gram[i][j]=ans        
        return Gram
    
    def cal(self,G,y_data,i):
        global a,b
        res=0
        for j in range(len(y_data)):
            res+=a[j]*y_data[j]*G[j][i]
        res+=b
        res*=y_data[i]
        return res
    
    def update(self,i,y):
        global a,b
        a[i]+=self.eta_
        b+=self.eta_*y

    def cal_wb(self,x_data,y_data):
        global a,b
        w=[0]*(x_data.shape[1])
        h = 0
        for i in range(len(y_data)):
            h +=a[i]*y_data[i]
            w +=a[i]*y_data[i]*x_data[i]
        print (w,h)
    
    def cal_w(self,x_data,y_data):
        global a,b
        w=[0]*(x_data.shape[1])
        for i in range(x_data.shape[1]):
            for j in range(len(y_data)):
                w[i]+=a[j]*y_data[j]*x_data[j][i]
        print(w,b)

if __name__=="__main__":
    x_data=np.array([[3,3],[4,3],[1,1]])
    y_data=[1,1,-1]
    clf=dualPerceptron(1)
    clf.fit(x_data,y_data)





