clc
clear all
close all

t1=clock;
ys_data=xlsread('data6.xlsx'); %����ԭʼ���ݣ���һ�з������

[n,p]=size(ys_data); %n�У�p��

for i=2:p
    Y=ys_data(:,1); %���������ֵ��Y
    X=ys_data(:,i); %������Ա�����ֵ��X
    xs(1,i-1)=i; %�������ϵ������ĵ�һ����ӱ���������λ�ñ��1��p
    xs(2,i-1)=abs(corr(X,Y,'type','Pearson')); %��������Ա�������������Pearson����ԣ�������xs����ڶ���
end
[n_xs,id]=sort(xs(2,:)); %�������������ϵ����С�����������У��洢��n_xs������
nid_xs=xs(1,id); %�������ϵ���������е�id�洢��nid_xs�У��������������λ�����������ϵ���������б���
for i=1:11
    xyb(:,i+1)=ys_data(:,nid_xs(i)); %���������ϵ��λ��ǰ ���ı������������������µ���������
end
xyb(:,1)=Y;

xlswrite('����������.xlsx',xyb);%������������xybд�����������ݱ��

t2=clock;
etime(t2,t1)