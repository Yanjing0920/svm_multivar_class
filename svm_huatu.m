clc
clear all
close all

% t1=clock;
% data=xlsread('data.xlsx','A1:Q473');
data=xlsread('新样本数据.xlsx');
label=xlsread('label.xlsx','A1:A473');

% figure;
% boxplot(data,'orientation','horizontal','labels',categories);
% title('细胞数据的box可视化图','FontSize',12);
% xlabel('属性值','FontSize',12);
% grid on;

% 画出测试数据的分维可视化图
% figure
% subplot(6,3,1);
% hold on;
% for run = 1:473
%     plot(run,label(run),'.');
% end
% xlabel('Samples No.','FontSize',10);
% ylabel('Label','FontSize',10);
% title('class','FontSize',10);
% set(gca,'YTick',[1:1:2]);
% for run = 2:18
%     subplot(6,3,run);
%     hold on;
%     str = ['Column-',num2str(run-1)];
%     for i = 1:473
%         plot(i,data(i,run-1),'.');
%     end
%     xlabel('Samples No.','FontSize',10);
%     ylabel('Value','FontSize',10);
%     title(str,'FontSize',10);
% end

% 选定训练集和测试集
A=data(1:242,:);
B=data(243:473,:);
len1=125;
len2=100;
A1=A(randperm(242,242),:);
B1=B(randperm(231,231),:);
train_A=A1(1:len1,:);
train_B=B1(1:len1,:);
% t2=clock;
% e1=etime(t2,t1)

A2=A(len1+1:242,:);
B2=B(len1+1:231,:);
A3=A2(randperm(size(A2,1),size(A2,1)),:);
B3=B2(randperm(size(B2,1),size(B2,1)),:);
test_A=A3(1:len2,:);
test_B=B3(1:len2,:);

% t3=clock;
trainData=[train_A;train_B];%选取训练数据
train_label=[label(1:len1);label(243:242+len1)]; %选取训练数据类别标识
% t4=clock;
% e2=etime(t4,t3)
testData=[test_A;test_B]; %选取测试数据
test_label=[label(1:len2);label(243:242+len2)]; %选取测试数据类别标识

% <span style='font-size:12px;'>%% SVM网络训练
% t5=clock;
cmd='-t 2 -c 0.5 -g 0.6'; %设置参数
model=libsvmtrain(train_label,trainData,cmd); %模型文件
% t6=clock;
% e3=etime(t6,t5)

 
% % SVM网络预测
t1=clock;
[predict_label,accuracy,prob_values] = libsvmpredict(test_label,testData,model);
t2=clock;
etime(t2,t1)

% 结果分析
 
% 测试集的实际分类和预测分类图
% 通过图可以看出只有多少个测试样本是被错分
figure;
hold on;
m=(test_label~=predict_label);
index=[];
for i=1:200
    if m(i,1)==1
        index=[index;i];
    end
end
r=size(index,1);
index0=zeros(r,1);
for j=1:r
    index0(j,1)=1;
end
index=[index index0];
predict_label(index(:,1))=nan;

plot(predict_label,'r*');
plot(index(:,1),index(:,2),'ko');
legend('Predict the correct sample point','Predict the wrong sample point');
xlabel('Serial number of testing samples');
set(gca,'XTick',[0:100:200]);
set(gca,'YTick',[1:1:2]);
grid on;

% t2=clock;
% etime(t2,t1)
