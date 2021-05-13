#!python
'''
Original Perceptron
'''
import csv
import numpy as np
import matplotlib.pyplot as plt
import math
import sys


# ============ FILE load and write stuff ===========================
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def read_asc_data(filename):    
    f= open(filename,'r') 
    tmp_str=f.readline()
    tmp_arr=tmp_str[:-1].split(' ')
    N=int(tmp_arr[0]);n_row=int(tmp_arr[1]);n_col=int(tmp_arr[2])
    data=np.zeros([N,n_row*n_col+1])
    for n in range(N):
        tmp_str=f.readline()
        tmp_arr=tmp_str[:-1].split(' ')       
        for i in range(n_row*n_col+1):
            data[n][i]=int(tmp_arr[i])
    f.close() 
    return N,n_row,n_col,data

def plot_data(row,col,n_row,n_col,data):
    fig=plt.figure(figsize=(row,col))
    for n in range(1, row*col +1):
        img=np.reshape(data[n-1][:-1],(n_row,n_col))
        fig.add_subplot(row, col, n)
        plt.imshow(img,interpolation='none',cmap='binary')
    plt.show()
    
def plot_tagged_data(row,col,n_row,n_col,X,Y,ew): 
    fig=plt.figure(figsize=(row,col))
    for n in range(row*col):
        img=np.reshape(X[n],(n_row,n_col))
        fig.add_subplot(row, col, n+1)
        #if(Y[n]>0):#exact case
        yn,a=predictor(X[n],ew)
        if(yn>0):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
    
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,1])
    plt.show()
    return 

def confusion(Xeval,Yeval,N,ew):
    C=np.zeros([2,2])
    for n in range(N):
        y,r=predictor(Xeval[n],ew)
        if(y<0. and Yeval[n]<0.): C[0,0]=C[0,0]+1
        if(y>0. and Yeval[n]>0.): C[1,1]=C[1,1]+1
        if(y<0. and Yeval[n]>0.): C[1,0]=C[1,0]+1
        if(y>0. and Yeval[n]<0.): C[0,1]=C[0,1]+1
    return C
#============== Perceptron Stuff ==================
def normalization(ew):
    return(ew/np.linalg.norm(ew,2))

def predictor(x,ew):
    r=ew[0]
    r=r+np.dot(x,ew[1:])
    sgn=np.sign(r)
    return sgn,r

def cost(X,Y,N,ew):
    En=0
    for n in range(N):
        ypred,a=predictor(X[n],ew)
        En=En+np.abs(0.5*(ypred-Y[n]))
    En=En/N
    return En

def update(x,y,eta,ew):
    ypred,a=predictor(x,ew)
    cor=1/(1+math.fabs(a)/eta)
    r=eta*0.5*(y-ypred)*cor
    ew[0]=ew[0]+r
    ew[1:]=ew[1:]+r*x
    return ew

def run_stocastic(X,Y,N,subloop,eta,MAX_ITER,ew,err):
    epsi=0
    it=0
    while(err[-1]>epsi):
        for j in range(subloop):
            n=int(np.random.rand()*N)
            ew=update(X[n],Y[n],eta,ew)
        ew=normalization(ew)
        EN=cost(X,Y,N,ew)
        err.append(EN)
        print('iter %d, cost=%f, eta=%e \r' % (it,EN,eta),end='')
        it=it+subloop   
        if(it>MAX_ITER): break
    #ew=normalization(ew);
    return ew, err


#=========== MAIN CODE ===============
N,n_row,n_col,data=read_asc_data('./Data/line600.txt')
#N,n_row,n_col,data=read_asc_data('./Data/line1500.txt')
#N,n_row,n_col,data=read_asc_data('./Data/rectangle600.txt')
#N,n_row,n_col,data=read_asc_data('./Data/square_circle.txt')
#N,n_row,n_col,data=read_asc_data('./Data/XOR.txt')
#N,n_row,n_col,data=read_asc_data('./Data/AND.txt')
print('find %d images of %d X %d pixels' % (N,n_row,n_col))
#plot_data(10,10,n_row,n_col,data)
Nt=int(N*0.8);I=n_row*n_col; #split training vs test sets
Xt=data[:Nt,:-1];Yt=data[:Nt,-1]
np.place(Yt, Yt!=1, [-1])
ew=np.ones([I+1]);ew=normalization(ew); #initialisation
err=[];err.append(cost(Xt,Yt,Nt,ew))
print('cost=%f ' % (err[-1]))
# ======= RUNNING ===========================
eta=0.5;nbiter=5000;subloop=50
ew,err=run_stocastic(Xt,Yt,Nt,subloop,eta,nbiter,ew,err)
print('cost=%f                ' % (err[-1]));eta=0.4*eta
ew,err=run_stocastic(Xt,Yt,Nt,subloop,eta,nbiter,ew,err)
print('cost=%f                ' % (err[-1]));eta=0.3*eta
ew,err=run_stocastic(Xt,Yt,Nt,subloop,eta,nbiter,ew,err)
print('cost=%f                ' % (err[-1]));eta=0.3*eta
ew,err=run_stocastic(Xt,Yt,Nt,subloop,eta,nbiter,ew,err)
print('cost=%f                ' % (err[-1]));eta=0.5*eta
ew,err=run_stocastic(Xt,Yt,Nt,subloop,eta,nbiter,ew,err)
print('cost=%f                ' % (err[-1]));eta=0.5*eta
plot_error(err)
# ============ OUTPUT =====================
#--- In-samples error ---
print
print('=========')
print('in-samples error=%f' % (err[-1]))
C =confusion(Xt,Yt,Nt,ew)
print(C)
#--- Out-samples error ---
Ne=N-Nt
Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1]
np.place(Ye, Ye!=1, [-1])
print('--------')
print('out-samples error=%f' % (cost(Xe,Ye,Ne,ew)))
C =confusion(Xe,Ye,Ne,ew)
print(C)
plot_tagged_data(10,10,n_row,n_col,Xe,Ye,ew)
print('bye')


