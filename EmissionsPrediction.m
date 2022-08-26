function network=LM(x)
nhidden=[26];
nhidden(nhidden==0) = [];
load SOxGas3
%normalized
input=SOxGas3(1:4532,1:5);
d=SOxGas3(1:4532,6);
[inputn, xs]=mapminmax(input');
[dn,ds]=mapminmax(d');
net=feedforwardnet(nhidden);

for i=1:length(nhidden)+1
    net.layers{i}.transferFcn = 'tansig';
    if i==length(nhidden)+1
       net.layers{i}.transferFcn = 'purelin';
    end
end
%train
net.trainFcn = 'trainlm';
%divide train
net.divideParam.trainRatio = 85/100;
net.divideParam.valRatio = 7.5/100;
net.divideParam.testRatio = 7.5/100;
%performance

net.performFcn = 'mse';
%adaption learning function
for j=1:length(nhidden)+1
    if j==1
     net.inputWeights{j,1}.learnFcn='learngd';
    end
     if j>1
     net.layerWeights{j,j-1}.learnFcn='learngd';
     end
     net.biases{j}.learnFcn='learngd';
end
%stopping criteria
net.trainParam.epochs=1000;
net.trainParam.goal=0;
net.trainParam.max_fail=12;
net.trainParam.min_grad=1e-8;
net.trainParam.mu=0.001;
net.trainParam.mu_dec=0.1;
net.trainParam.mu_inc=10;
net.trainParam.mu_max=1e10;
net.trainParam.time=inf;

%train
[net,tr]=train(net,inputn,dn);
%save
SOxGas3output=net;
save ('C:\Users\bildiran\Desktop\qom power plant\SOx 3 - Gas\SOxGas3output')
%output weihts
w=cell((length(nhidden)+1),1);
wb=cell((length(nhidden)+1),1);
for j1=1:(length(nhidden)+1)
    if j1==1
     w{j1}=net.IW{j1};
    end
     if j1>1
     w{j1}=net.LW{j1,j1-1};
     end
     wb{j1}=net.b{j1};
end
%save

ve=[]
bper=round([tr.best_perf,tr.best_vperf,tr.best_tperf],5)
%network.weights=w
%network.weightsbias=wb
%network.characteristic=tr;
%network.performance=bper;
an=sim(net,inputn(:,tr.testInd));
a = mapminmax('reverse',an,ds);
%validation
van=sim(net,inputn(:,tr.valInd));
va = mapminmax('reverse',van,ds);
%train

tan=sim(net,inputn(:,tr.trainInd));
ta = mapminmax('reverse',tan,ds);
output1=d';
ttest=output1(:,tr.testInd);
[r] = regression(ttest,a);
%validation
vtest=output1(:,tr.valInd);
[rv] = regression(vtest,va);
%train
trtest=output1(:,tr.trainInd);
[rt] = regression(trtest,ta);
%APE test
APE=[];
for i=1:length(a)
APE(i)=abs((a(i)-ttest(i))/ttest(i));
end
meanAPE=round(mean(APE),5);
maxAPE=round(max(APE),5);
minAPE=round(mean(APE),5);
%output vector
ve=[bper(1),bper(2),bper(3),round(rt,5),round(rv,5),round(r,5),tr.num_epochs,minAPE,maxAPE,meanAPE];
network=[w wb];
%network.net=net;
end
