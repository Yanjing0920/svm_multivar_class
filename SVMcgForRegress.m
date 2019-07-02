function [mse,bestc,bestg] = SVMcgForRegress(train_label,train,cmin,cmax,gmin,gmax,v,cstep,gstep,msestep)
%SVMcg cross validation by faruto


% faruto and liyang , LIBSVM-farutoUltimateVersion 
% a toolbox with implements for support vector machines based on libsvm, 2009. 
% Software available at http://www.ilovematlab.cn
% 
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%about the parameters of SVMcg 
if nargin < 10
    msestep = 0.03;
end
if nargin < 8
    cstep = 0.8;
    gstep = 0.8;
end
if nargin < 7
    v = 10;
end
if nargin < 5
    gmax = 8;
    gmin = -8;
end
if nargin < 3
    cmax = 8;
    cmin = -8;
end
%X:c Y:g cg:acc
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);

eps = 10^(-4);

%record acc with different c & g,and find the bestacc with the smallest c
bestc = 0;
bestg = 0;
mse = Inf;
basenum = 2;
for i = 1:m
    for j = 1:n
        cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) ),' -s 3 -p 0.1 -h 0'];
        cg(i,j) = libsvmtrain(train_label, train, cmd);
        
        if cg(i,j) < mse
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
        if abs( cg(i,j)-mse )<=eps && bestc > basenum^X(i,j)
            mse = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end
        
    end
end
%to draw the acc with different c & g
[cg,ps] = mapminmax(cg,0,1);
figure;
[C,h] = contour(X,Y,cg,0.70:msestep:0.975);
clabel(C,h,'FontSize',7,'Color','r');
% set(h,'ShowText','on','LevelList',[1,2,3,4,5,6,7,8]);
% set(h,'ShowText','off');
xlabel('logg','FontSize',12);
ylabel('logc','FontSize',12);
% firstline = 'Result of parameters selection'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVmse=',num2str(mse)];
% title({firstline;secondline},'Fontsize',12);
grid on;
figure;
meshc(X,Y,cg);
% mesh(X,Y,cg);
% surf(X,Y,cg);
axis([cmin,cmax,gmin,gmax,0,1]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('MSE','FontSize',12);
% firstline = 'Result of parameters selection(3D)'; 
% secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
%     ' CVmse=',num2str(mse)];
% title({firstline;secondline},'Fontsize',12);