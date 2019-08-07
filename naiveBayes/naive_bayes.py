'''
朴素贝叶斯分类算法
Step1: 采用贝叶斯估计方法计算先验概率分布
Step2: 采用贝叶斯估计方法计算条件概率分布
Step3：根据Step1, 2结果，计算联合概率分布

lambda_取值为0时，为极大似然估计
lambda_取值为1时，为拉普拉斯平滑估计
'''

import numpy as np
from collections import Counter,defaultdict
import xlrd

class naiveBayes:
    def __init__(self,lambda_=1):
        self.lambda_=lambda_
        self.p_prior={}
        self.p_condition={}
    
    def fit(self,x_data,y_data):
        #计算先验概率
        N=y_data.shape[0]#数据集Y的长度
        count_y=Counter(y_data)
        K=len(count_y)#Y取值的种类数目
        for key,value in count_y.items():
            self.p_prior[key]=(value+self.lambda_)/(N+K*self.lambda_)

        #计算条件概率
        for j in range(x_data.shape[1]):
            Xj_y=defaultdict(int)
            Xj=x_data[:,j]#提取X的某一特征J的取值
            Sj=len(np.unique(Xj))#特征J取值的种类数目
            for xj,y in zip(Xj,y_data):
                Xj_y[(xj,y)]+=1
            for key,value in Xj_y.items():
                self.p_condition[(j,key[0],key[1])]=(value+self.lambda_)/(count_y[key[1]]+Sj*self.lambda_)
            for m in np.unique(Xj):
                for n in np.unique(y_data):
                    if self.p_condition.get((j,m,n))==None:
                        self.p_condition[(j,m,n)]=self.lambda_/(count_y[key[1]]+Sj*self.lambda_)

    def predict_point(self,X):
        #输入为一个实例
        #计算联合概率        
        p_post=defaultdict()
        for y,py in self.p_prior.items():
            p_joint=py
            for j,xj in enumerate(X):           
                p_joint*=self.p_condition[(j,xj,y)]#条件独立性假设
            p_post[y]=p_joint#后验概率（忽略分母）
        return max(p_post,key=p_post.get)
    
    def predict(self,X):
        #输入为实例集合
        post=[]
        if len(X.shape)==1:
            post.append(self.predict_point(X))
        else:
            for j in range(X.shape[0]):
                Xj=X[j,:]
                post.append(self.predict_point(Xj))
        return post

def load_txt(path,sep='\t'):
    data=[]
    f=open(path,encoding='utf-8')    
    for line in f:
        sample=[]
        for item in line.strip.split(sep):
            sample.append(item)
        data.append(sample)   
    return data
def load_excel(path):
    data=[]
    f=xlrd.open_workbook(path)    
    Data_sheet=f.sheets()[0]
    #print(Data_sheet.name)
    rowNum=Data_sheet.nrows
    colNum=Data_sheet.ncols
    #print(rowNum,colNum)
    for i in range(rowNum):
        rowlist=[]
        for j in range(colNum):
            cvalue=Data_sheet.cell_value(i,j)
            ctype=Data_sheet.cell(i,j).ctype
            if ctype==2 and cvalue%1==0:
                cvalue=int(cvalue)
            rowlist.append(cvalue)
        data.append(rowlist)
    data=data[1:]   
    return data

if __name__=='__main__':
    #数据处理
    '''
    data = np.array([[1, 'S', -1], [1, 'M', -1], [1, 'M', 1], [1, 'S', 1],
                     [1, 'S', -1], [2, 'S', -1], [2, 'M', -1], [2, 'M', 1],
                     [2, 'L', 1], [2, 'L', 1], [3, 'L', 1], [3, 'M', 1],
                     [3, 'M', 1], [3, 'L', 1], [3, 'L', -1]])
    '''
    path=r'E:\PYproject\data\eg4.1.xlsx'
    data=load_excel(path)
    #data=load_txt(path)
    data=np.array(data)   
    #print(data)
    X_data=data[:,:-1]
    y_data=data[:,-1]

    #训练
    clf=naiveBayes(lambda_=1)
    clf.fit(X_data,y_data)
    #print(clf.p_prior)
    #print(clf.p_condition)

    #预测
    #x=np.array([2,'S'])
    x= np.array([[1, 'S'], [1, 'M'], [1, 'M']])
    result=clf.predict(x)
    print(result)

