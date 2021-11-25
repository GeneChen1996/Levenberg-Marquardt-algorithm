clear
clc
%% 資料整理 變數設定
load('iris.txt')
dataSet = load('iris.txt');
x = dataSet([51:150],[3,4]);
y = [ones(1,50),ones(1,50)-1]';
x=[ones(length(y),1) x];
L = zeros(10, 1);

m = length(y);
w = zeros(3,1);
sigmoid = inline('1./ (1.0 + exp(-z))'); %1./ (1.0 + exp(-z))       exp(z)./ (1.0 + exp(z))
pos = find(y == 1);
neg = find(y == 0);
S = find(y==1 | y==0);

%% Newton' method 迭代10次¸
for iterations = 1:10
    P = sigmoid(x * w);   
   
    L(iterations) = sum(-y.*log(P)-(1-y).*log(1- P(S)))/m;
    
    delta_L = sum(repmat((y-P),1,3).*x);
    
    F = x'*(repmat((P.*(1-P)),1,3).*x);
    
    w = w + inv(F)*delta_L';
end

%% 畫圖
figure
plot(0:9, L, 'o-')
xlabel('Number of iterations')
ylabel('Cost L')

figure
plot(x(pos, 2), x(pos,3), '+'); hold on
plot(x(neg, 2), x(neg, 3), 'o')
plot(x(:,2), (-w(1).*x(:,1) - w(2).*x(:,2))/w(3), 'v');
legend('classP', 'classN', 'Decision Boundary');
xlabel('featureA')
ylabel('featureB')

hold off


