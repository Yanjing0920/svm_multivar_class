clc
clear all
close all

t1=clock;
ys_data=xlsread('data6.xlsx'); %读入原始数据，第一列放因变量

[n,p]=size(ys_data); %n行，p列

for i=2:p
    Y=ys_data(:,1); %将因变量赋值给Y
    X=ys_data(:,i); %逐个将自变量赋值给X
    xs(1,i-1)=i; %将相关性系数数组的第一行添加变量所在列位置标记1：p
    xs(2,i-1)=abs(corr(X,Y,'type','Pearson')); %逐个计算自变量与因变量间的Pearson相关性，保存至xs数组第二行
end
[n_xs,id]=sort(xs(2,:)); %将变量间相关性系数按小到大升序排列，存储在n_xs矩阵中
nid_xs=xs(1,id); %将相关性系数升序排列的id存储于nid_xs中，即保存变量所在位置随着相关性系数降序排列保存
for i=1:11
    xyb(:,i+1)=ys_data(:,nid_xs(i)); %保存相关性系数位于前 个的变量，并和因变量组成新的样本数据
end
xyb(:,1)=Y;

xlswrite('新样本数据.xlsx',xyb);%将新样本数据xyb写到新样本数据表格

t2=clock;
etime(t2,t1)