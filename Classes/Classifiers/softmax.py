#!python
'''
softmax_classifier
'''
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
import math

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
    print("N=%d, row=%d, col=%d" %(N,n_row,n_col))
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
        if(predictor(X[n],ew)>0.5):
            plt.imshow(img,interpolation='none',cmap='RdPu')
        else:
            plt.imshow(img,interpolation='none',cmap='cool')               
    plt.show()
    
def plot_error(err):
    plt.plot(range(len(err)), err, marker='o')
    plt.xlabel('Iterations')
    plt.ylabel('Number of misclassifications')
    plt.ylim([0,5])
    plt.show()
    return 

def confusion(Xeval,Peval,N,J,W):
    C=np.zeros([J,J]);
    for n in range(N):
        p=predictor(Xeval[n],W);
        i=np.argmax(Peval[n]);
        j=np.argmax(p);
        C[i,j]=C[i,j]+1;
    return C

def softmax(s):
    s=np.maximum(s,-30);
    s=np.minimum(s,30);
    sm =np.exp(s)/np.sum(np.exp(s), axis=0)
    return sm
     

def convert_probability(Yt,Nt,J):
    Pt=np.zeros([Nt,J])
    for n in range(Nt):
        i=int(Yt[n])
        Pt[n,i]=1;
    return Pt
#============== Logistic classifier Stuff ==================


def predictor(x,W):


def cost(X,P,N,W):


def update(x,p,eta,W):



def run_stocastic(X,P,N,eta,MAX_ITER,W,err):




#=========== MAIN CODE ===============
# read the data file
#N,n_row,n_col,data=read_asc_data('./AND.txt')
#N,n_row,n_col,data=read_asc_data('./XOR.txt')
#N,n_row,n_col,data=read_asc_data('./line600.txt')
N,n_row,n_col,data=read_asc_data('./line1500.txt')
#N,n_row,n_col,data=read_asc_data('./my_digit.txt');
random.shuffle(data)
print('find %d images of %d X %d pixels' % (N,n_row,n_col))
J=len(np.unique(data[:N,-1]));print("number of class values:",J)

Nt=int(N*0.5);
I=n_row*n_col;
Xt=data[:Nt,:-1];Yt=data[:Nt,-1];Pt=convert_probability(Yt,Nt,J)
W=np.ones([J,I+1]); err=[];err.append(cost(Xt,Pt,Nt,W));

W,err=run_stocastic(Xt,Pt,Nt,1.0,2499,W,err);print("\n")
W,err=run_stocastic(Xt,Pt,Nt,0.1,1999,W,err);print("\n")
W,err=run_stocastic(Xt,Pt,Nt,0.03,1999,W,err);print("\n")

plot_error(err)
print("\n\n");
print('in-samples error=%f' % (err[-1]))
C =confusion(Xt,Pt,Nt,J,W)
print(C)

Ne=N-Nt;Xe=data[Nt:N,:-1];Ye=data[Nt:N,-1];Pe=convert_probability(Ye,Ne,J)
print('out-samples error=%f' % (cost(Xe,Pe,Ne,W)))
C =confusion(Xe,Pe,Ne,J,W)
print(C)

print('bye')