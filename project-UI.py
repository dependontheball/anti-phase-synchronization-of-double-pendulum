from dataclasses import InitVar
from tkinter import *
import math
from pyexpat.model import XML_CTYPE_EMPTY
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

root = Tk()
m_pre=DoubleVar()
M_pre=DoubleVar()
g_pre=DoubleVar()
l_pre=DoubleVar()
mu_pre=DoubleVar()
q1_pre=DoubleVar()
q2_pre=DoubleVar()
frame_pre=DoubleVar()
h_pre=DoubleVar()
arrey=[1,1,1,1,1,1,1,1,1]

toptitle_1 = Label(root,text="define the vaule of variables",fg="black",font=20).grid(row=0,column=0)
toptitle_2 = Label(root,text="                             ",fg="black",font=20).grid(row=0,column=1)
toptitle_3 = Label(root,text="                                        ",fg="black",font=20).grid(row=0,column=2)
#def m_define():
m1=Label(root,text="m : ",fg="black",font=20)
m1.grid(row=1,column=0)
m2=Entry(textvariable=m_pre)
m2.grid(row=1,column=1)
m2.delete(0,END)
m2.insert(0,0.7)
#def M_define():
M1=Label(root,text="M : ",fg="black",font=20)
M1.grid(row=2,column=0)
M2=Entry(textvariable=M_pre)
M2.grid(row=2,column=1)
M2.delete(0,END)
M2.insert(0,0.5)
#def g_define():
g1 = Label(root,text="g : ",fg="black",font=20)
g1.grid(row=3,column=0)
g2=Entry(textvariable=g_pre)
g2.grid(row=3,column=1)
g2.delete(0,END)
g2.insert(0,9.8)
#def l_define():
l1 = Label(root,text="l : ",fg="black",font=20)
l1.grid(row=4,column=0)
l2=Entry(textvariable=l_pre)
l2.grid(row=4,column=1)
l2.delete(0,END)
l2.insert(0,0.25)    
#def mu_define():
mu1 = Label(root,text="mu : ",fg="black",font=20)
mu1.grid(row=5,column=0)
mu2=Entry(textvariable=mu_pre)
mu2.grid(row=5,column=1)
mu2.delete(0,END)
mu2.insert(0,0.9) 
#def q1_define():
q_one_1= Label(root,text="q1 : ",fg="black",font=20)
q_one_1.grid(row=6,column=0)
q_one_2=Entry(textvariable=q1_pre)
q_one_2.grid(row=6,column=1)
q_one_2.delete(0,END)
q_one_2.insert(0,72)
#def q2_define():
q_two_1= Label(root,text="q2 : ",fg="black",font=20)
q_two_1.grid(row=7,column=0)
q_two_2=Entry(textvariable=q2_pre)
q_two_2.grid(row=7,column=1)
q_two_2.delete(0,END)
q_two_2.insert(0,-10)
#def frame_pre_define():
frame1= Label(root,text="frame : ",fg="black",font=20)
frame1.grid(row=8,column=0)
frame2=Entry(textvariable=frame_pre)
frame2.grid(row=8,column=1)
frame2.delete(0,END)
frame2.insert(0,500000)
#def h_pre_define():
h1= Label(root,text="h : ",fg="black",font=20)
h1.grid(row=9,column=0)
h2=Entry(textvariable=h_pre)
h2.grid(row=9,column=1)
h2.delete(0,END)
h2.insert(0,0.00001)

def mainprocess():
    save()
    #define solution
    q1D = 0
    q2D = 0
    m=arrey[0]
    M=arrey[1]
    g=arrey[2]
    l=arrey[3]
    mu=arrey[4]
    q1=arrey[5]
    q2=arrey[6]
    frame=int(arrey[7])
    h=float(arrey[8])
    q1=np.sin(np.radians(q1))
    q2=np.sin(np.radians(q2))
    K=-g*math.tan(q1)
    J=-g*math.tan(q2)
    A=l*((1/math.cos(q1))-((m/M+2*m)*math.cos(q1)))
    B=l*((1/math.cos(q2))-((m/M+2*m)*math.cos(q2)))
    C=-(math.cos(q2)*m*l)/(M+2*m)
    D=-(math.cos(q2)*m*l)/(M+2*m)
    q1DD =(K*B-J*C)/(A*B-C*D)
    q2DD =(J*A-D*K)/(A*B-C*D)

    K=-g*math.tan(q1)
    J=-g*math.tan(q2)
    A=l*((1/math.cos(q1))-((m/M+2*m)*math.cos(q1)))
    B=l*((1/math.cos(q2))-((m/M+2*m)*math.cos(q2)))
    C=-(math.cos(q2)*m*l)/(M+2*m)
    D=-(math.cos(q2)*m*l)/(M+2*m)

    q1DD =(K*B-J*C)/(A*B-C*D)
    q2DD =(J*A-D*K)/(A*B-C*D)

    print(q1DD)
    print(q2DD)

    x=0
    xD=10**-36
    F= -mu*M*g*xD/abs(xD)
    xDD= -(m*l)/(M+2*m) * (q1DD*np.cos(q1) + q2DD*np.cos(q2) - (q1D**2)*np.sin(q1) - (q2D**2)*np.sin(q2)) + F/(M+2*m)

    p = (M + 2*m) * xD + m * l * (q1D*np.cos(q1) + q2D*np.cos(q2)) #linear momentum

    t=0

    tx=[]

    q1Dz=[]
    q2Dz=[]

    xz=[]
    x1z=[]
    x2z=[]

    xDz=[]
    x1Dz=[]
    x2Dz=[]

    xcmz=[]
    vcmz=[]



    for i in range(0,frame):

        F= -mu*M*g*xD/abs(xD)

        xDD = -(m*l)/(M+2*m) * (q1DD*np.cos(q1) + q2DD*np.cos(q2) - (q1D**2)*np.sin(q1) - (q2D**2)*np.sin(q2)) + F/(M+2*m)
        xD = xD + xDD*h
        x = x + xD*h + 1/2*xDD*h**2

        q1DD = -1/l*(g*np.sin(q1) + xDD*np.cos(q1))
        q1D = q1D + q1DD*h
        q1 = q1 + q1D*h + 1/2*q1DD*h**2

        q2DD = -1/l*(g*np.sin(q2) + xDD*np.cos(q2))
        q2D = q2D + q2DD*h
        q2 = q2 + q2D*h + 1/2*q2DD*h**2
        
        x1 = l*math.sin(q1)
        x2 = l*math.sin(q2)

        x1D = xD + l*math.cos(q1)*q1D
        x2D = xD + l*math.cos(q2)*q2D

        vcm = (m*x1D + m*x2D + M*xD)/(M + 2*m)

        t = t + h
        
        tx.append(t)

        q1Dz.append(q1D)
        q2Dz.append(q2D)

        xz.append(x)
        x1z.append(x1)
        x2z.append(x2)

        xDz.append(xD)
        x1Dz.append(x1D)
        x2Dz.append(x2D)

        vcmz.append(vcm)

    print(t)

    fig = plt.gcf()
    plt.plot(tx,x1Dz,color='r',label='1st pendulum')
    plt.plot(tx,x2Dz,color='g',label='2nd pendulum')
    plt.plot(tx,xDz,color='orange',label='tie mass')
    plt.plot(tx,vcmz,color='c',label='CM')
    plt.ylabel('velocity')
    plt.xlabel('time')
    plt.title('doubleOsSyn')
    plt.legend()
    fig.set_size_inches(20, 6)
    plt.show()

    fig = plt.gcf()
    plt.plot(tx,x1z,color='r',label='1st pendulum')
    plt.plot(tx,x2z,color='g',label='2nd pendulum')
    plt.plot(tx,xz,color='orange',label='tie mass')
    plt.ylabel('position')
    plt.xlabel('time')
    plt.title('doubleOsSyn')
    plt.legend()
    fig.set_size_inches(20, 6)
    plt.show()

def save():
    for i in range(0,9):
        arrey[i]=0
    arrey[0]=m_pre.get()
    arrey[1]=M_pre.get()
    arrey[2]=g_pre.get()
    arrey[3]=l_pre.get()
    arrey[4]=mu_pre.get()
    arrey[5]=q1_pre.get()
    arrey[6]=q2_pre.get()
    arrey[7]=frame_pre.get()
    arrey[8]=h_pre.get()
    print(arrey)
    return arrey
    
def reset():
    m2.delete(0,END)
    m2.insert(0,0.7)
    M2.delete(0,END)
    M2.insert(0,0.5)
    g2.delete(0,END)
    g2.insert(0,9.8)
    l2.delete(0,END)
    l2.insert(0,0.25)
    mu2.delete(0,END)
    mu2.insert(0,0.9)
    q_one_2.delete(0,END)
    q_one_2.insert(0,72)
    q_two_2.delete(0,END)
    q_two_2.insert(0,-10)
    frame2.delete(0,END)
    frame2.insert(0,500000)
    h2.delete(0,END)
    h2.insert(0,0.00001)

def clear():
    m2.delete(0,END)
    M2.delete(0,END)
    g2.delete(0,END)
    l2.delete(0,END)
    mu2.delete(0,END)
    q_one_2.delete(0,END)
    q_two_2.delete(0,END)
    frame2.delete(0,END)
    h2.delete(0,END)

btn2=Button(text="reset",command=reset)
btn2.grid(row=10,column=1)
btn3=Button(text="Clear",command=clear)
btn3.grid(row=10,column=2)
btn3=Button(text="plot",command=mainprocess)
btn3.grid(row=10,column=0)

#def h_pre_define():
info1= Label(root,text=" ",fg="black",font=20)
info1.grid(row=12,column=0)


#กำหนดหน้าจอและตำแหน่งหน้าจอ
root.geometry("500x400+100+0")

root.mainloop()