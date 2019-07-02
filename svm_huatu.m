clc
clear all
close all

% t1=clock;
% data=xlsread('data.xlsx','A1:Q473');
data=xlsread('����������.xlsx');
label=xlsread('label.xlsx','A1:A473');

% figure;
% boxplot(data,'orientation','horizontal','labels',categories);
% title('ϸ�����ݵ�box���ӻ�ͼ','FontSize',12);
% xlabel('����ֵ','FontSize',12);
% grid on;

% �����������ݵķ�ά���ӻ�ͼ
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

% ѡ��ѵ�����Ͳ��Լ�
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
trainData=[train_A;train_B];%ѡȡѵ������
train_label=[label(1:len1);label(243:242+len1)]; %ѡȡѵ����������ʶ
% t4=clock;
% e2=etime(t4,t3)
testData=[test_A;test_B]; %ѡȡ��������
test_label=[label(1:len2);label(243:242+len2)]; %ѡȡ������������ʶ

% <span style='font-size:12px;'>%% SVM����ѵ��
% t5=clock;
cmd='-t 2 -c 0.5 -g 0.6'; %���ò���
model=libsvmtrain(train_label,trainData,cmd); %ģ���ļ�
% t6=clock;
% e3=etime(t6,t5)

 
% % SVM����Ԥ��
t1=clock;
[predict_label,accuracy,prob_values] = libsvmpredict(test_label,testData,model);
t2=clock;
etime(t2,t1)

% �������
 
% ���Լ���ʵ�ʷ����Ԥ�����ͼ
% ͨ��ͼ���Կ���ֻ�ж��ٸ����������Ǳ����
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
