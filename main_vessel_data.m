% This script provides an application example
% of the PAWN sensitivity analysis approach (Pianosi and Wagener, 2015)
%
% MODEL AND STUDY AREA
%
% The model under study is the rainfall-runoff model Hymod
% (see help of function hymod_sim.m for more details) 
% applied to the Leaf catchment in Mississipi, USA
% (see header of file LeafCatch.txt for more details).
% The inputs subject to SA are the 5 model parameters, and the scalar 
% output for SA is a statistic of the simulated time series
% (e.g. the maximum flow over the simulation horizon)
% 
% REFERENCES
%
% Pianosi, F. and Wagener, T. (2015), A simple and efficient method 
% for global sensitivity analysis based on cumulative distribution 
% functions, Env. Mod. & Soft., 67, 1-11.

% This script prepared by Francesca Pianosi and Fanny Sarrazin
% University of Bristol, 2015
% mail to: francesca.pianosi@bristol.ac.uk

clc
clear all
close all

data=xlsread('新样本数据.xlsx');
label=xlsread('label.xlsx','A1:A473');

A=data(1:242,:);
B=data(243:473,:);
len1=80;
len2=125;
A1=A(randperm(242,242),:);
B1=B(randperm(231,231),:);
train_A=A1(1:len1,:);
train_B=B1(1:len1,:);
A2=A(len1+1:242,:);
B2=B(len1+1:231,:);
A3=A2(randperm(size(A2,1),size(A2,1)),:);
B3=B2(randperm(size(B2,1),size(B2,1)),:);
test_A=A3(1:len2,:);
test_B=B3(1:len2,:);
trainData=[train_A;train_B];%选取训练数据
train_label=[label(1:len1);label(243:242+len1)]; %选取训练数据类别标识
testData=[test_A;test_B]; %选取测试数据
test_label=[label(1:len2);label(243:242+len2)]; %选取测试数据类别标识
% distance=ship_data(:,2);
% heading=ship_data(:,3);
%generate training data by this function
% total_power=sum(ship_data(:,[5 8 11 14 17 20]),2);
% input_data=ship_data(:,[5 8 11 14 17 20]);
% data0=[distance,heading,input_data];
% data_nopower=[Yu_train_nopower,input_data];
% data1_nopower=[distance,heading,input_data];
% GAELM(data_nopower);
% ELM_modelling(data_nopower)
% index=randperm(10000,5001);
% [row,col]=size(trainData);
% M=col;
% Yu_train=Yu_train(index,:);
% Yu_train1=Yu_train1(index,:);
% Yu_train_nopower=Yu_train_nopower(index,:);
% Yu_train1_nopower=Yu_train1_nopower(index,:);
% Xu_train=Xu_train(index,:);

% [acc,bestc,bestg]=SVMcgForRegress(Yu_train0(index,:),Xu_train(index,:));
% cmd=['-s 3 -t  2', '-c', num2str(bestc), '-g',num2str( bestg),  '-p 0.01 -v 10'];
% model_SVM0=libsvmtrain(Yu_train0(index,:),Xu_train(index,:),cmd);
% [acc,bestc,bestg]=SVMcgForRegress(Yu_train0_nopower(index,:),Xu_train(index,:));
% cmd=['-s 3 -t  2', '-c', num2str(bestc), '-g',num2str( bestg),  '-p 0.01 -v 10'];
% model_SVM0_nopower=libsvmtrain(Yu_train0_nopower(index,:),Xu_train(index,:),cmd);
% 

[bestacc,bestc,bestg]=SVMcgForClass(train_label,trainData);
cmd=['-s 3 -t 2', '-c', num2str(bestc), '-g',num2str( bestg), '-p 0.01 -v 5'];
model=libsvmtrain(train_label,trainData,cmd);
[predict_label,accuracy,prob_values] = libsvmpredict(test_label,testData,model);

% [acc,bestc,bestd]=SVMcgForClass(train_label,trainData);
% cmd=['-s 3 -t 1', '-c', num2str(bestc), '-g',num2str( bestg),'-d 3 -p 0.01 -v 10'];
% model=libsvmtrain(train_label,trainData,cmd);
% [predict_label,accuracy,prob_values] = libsvmpredict(test_label,testData,model);


% [mse,bestc,bestg]=SVMcgForRegress(train_label,trainData);
% cmd=['-s 3 -t 2', '-c', num2str(bestc), '-g',num2str( bestg),  '-p 0.01 -v 10'];
% model_SVM1=libsvmtrain(Yu_train1,Xu_train,cmd)

% [acc,bestc,bestg]=SVMcgForRegress(Yu_train_nopower(index,:),Xu_train(index,:));
% cmd=['-s 3 -t  2', '-c', num2str(bestc), '-g',num2str( bestg),  '-p 0.01 -v 10'];
% model_SVM_nopower=libsvmtrain(Yu_train_nopower(index,:),Xu_train(index,:),cmd)
% [acc,bestc,bestg]=SVMcgForRegress(Yu_train1_nopower(index,:),Xu_train(index,:));
% cmd=['-s 3 -t  2', '-c', num2str(bestc), '-g',num2str( bestg),  '-p 0.01 -v 10'];
% model_SVM1_nopower=libsvmtrain(Yu_train1_nopower(index,:),Xu_train(index,:),cmd)
% RBF
% [model_RBF,inputps_RBF,outputps_RBF]=RBF_train(Xu_train,Yu_train);
% %BP
% [model_BP,inputps_BP,outputps_BP]=BP_train(Xu_train,Yu_train);····`····`············````·`·
% 
% %RELM
% obj=KELM();
% model_KELM=obj.train(Xu_train,Yu_train);····
%% Step 3: Apply PAWN················
% distrpar =  [ -1 1];
% n=10;
% %using ishigami functin to generate input data
% Xu = AAT_sampling('lhs',M,'unif',distrpar,600); 
% % Create input/output samples to estimate the conditional output CDFs:··`
% [ XX, xc ] = pawn_sampling('lhs',M,'unif',distrpar,n,600);
% for i=1:10
% arg1{1,1}='SVM'
% % pawn_index_SVM0(i,:)=PAWN_index(Xu,XX,model_SVM0,arg1);
% % pawn_index_SVM0_nopower(i,:)=PAWN_index(Xu,XX,model_SVM0_nopower,arg1);
% pawn_index_SVM(i,:)=PAWN_index(Xu,XX,model_SVM,arg1);
% % pawn_index_SVM_nopower(i,:)=PAWN_index(Xu,XX,model_SVM_nopower,arg1);
% pawn_index_SVM1(i,:)=PAWN_index(Xu,XX,model_SVM1,arg1);
% % pawn_index_SVM1_nopower(i,:)=PAWN_index(Xu,XX,model_SVM1_nopower,arg1);
% end
% % Copy_pawn_index_SVM0=mean(pawn_index_SVM0);
% % Copy_pawn_index_SVM0_nopower=mean(pawn_index_SVM0_nopower);
%  Copy_pawn_index_SVM=mean(pawn_index_SVM);
%  pawn_index_SVM=Copy_pawn_index_SVM./sum(Copy_pawn_index_SVM);
% % Copy_pawn_index_SVM_nopower=mean(pawn_index_SVM_nopower);
% Copy_pawn_index_SVM1=mean(pawn_index_SVM1);
% pawn_index_SVM_1=Copy_pawn_index_SVM1./sum(Copy_pawn_index_SVM1);

% Copy_pawn_index_SVM1_nopower=mean(pawn_index_SVM1_nopower);
% pawn_index_SVM=pawn_index_SVM./sum(pawn_index_SVM);
% arg{1,2}=PS_Xu;
%arg{1,3}=PS_Yu;
% arg{1,1}='RBF'; 
% pawn_index_RBF(1,:)=PAWN_index(Xu,XX,model_RBF,arg);
% % arg{1,2}=inputps_BP;
% %arg{1,3}=outputps_BP;
% arg{1,1}='BP';
% pawn_index_BP(1,:)=PAWN_index(Xu,XX,model_BP,arg);
% arg1{1,1}='KELM'
% pawn_index_KELM(1,:)=PAWN_index(Xu,XX,model_KELM,arg1);
% pawn_index_KELM=pawn_index_KELM./sum(pawn_index_KELM);
% Plot:
% figure 
% boxplot1(Pi,labelparams)
% 
% % Use bootstrapping to assess robustness of PAWN indices:
% stat = 'max' ; % statistic to be applied to KSs
% Nboot = 100  ; % number of boostrap resamples
% [ T_m, T_lb, T_ub ] = pawn_indices(Yu,YY,stat,[],Nboot);

% Plot:
% figure; boxplot1(T_m,labelparams,[],T_lb,T_ub)

% Convergence analysis:
% stat = 'max' ; % statistic to be applied to KSs
% NCb = [ NC/10 NC/2 NC ] ;
% NUb = [ NU/10 NU/2 NU ] ;
% 
% [ T_m_n, T_lb_n, T_ub_n ] = pawn_convergence( Yu, YY, stat, NUb, NCb,[],Nboot );
% NN = NUb+n*NCb ;
% figure; plot_convergence(T_m_n,NN,T_lb_n,T_ub_n,[],'no of evals',[],labelparams)

%% Step 4: Apply PAWN to sub-region of the output range

% Compute the PAWN index over a sub-range of the output distribution, for
% instance only output values above a given threshold
% thres = 50 ;
% [ T_m2, T_lb2, T_ub2 ]= pawn_indices( Yu, YY, stat,[], Nboot,[],'above',thres ) ;
% 
% % Plot:
% figure; boxplot1(T_m2,labelparams,[],T_lb2,T_ub2)


