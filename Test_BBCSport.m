clear,clc;
addpath('tSVD','proxFunctions','solvers','twist');
addpath('ClusteringMeasure', 'LRR', 'Nuclear_norm_l21_Algorithm', 'unlocbox');

load('my_BBCSport.mat'); 
fprintf('============================MSCVC for BBCSport============================\n');

cls_num = length(unique(gt));

X{1} = X1; X{2} = X2;
for v=1:2
    [X{v}]=NormalizeData(X{v});
end

lambda1 = 1; 
lambda2 = 0.06; 
dim_H = 150;

K = length(X); N = size(X{1},2);

for k=1:K
    H{k} = zeros(dim_H,N);
    Z{k} = zeros(N,N);
    P{k} = zeros(size(X{k},1),dim_H);
    W1{k} = zeros(dim_H,N);
    W2{k} = zeros(N,N);
    C{k} = zeros(N,N);
    Q{k} = zeros(dim_H,N);
    E{k} = zeros(dim_H,N); 
    Y1{k} = zeros(size(X{k},1),N); 
    Y2{k} = zeros(dim_H,N);
end

w1 = zeros(dim_H*N*K,1);
w2 = zeros(N*N*K,1);
q = zeros(dim_H*N*K,1);
g = zeros(N*N*K,1);
dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX1 = [dim_H, N, K];
sX2 = [N, N, K];

parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;

Isconverg = 0;epson = 1e-7;
iter = 0;
mu = 10e-5; max_mu = 10e10; pho_mu = 1.3;
rho = 0.0001; max_rho = 10e12; pho_rho = 1.3;
tic;

while(Isconverg == 0)
    fprintf('processing iter %d ...\n', iter+1);
    for k=1:K
        H_a = mu*(P{k}'*P{k}) + (mu+rho)*eye(dim_H);
        H_b = mu*(Z{k}*Z{k}'-Z{k}-Z{k}');
        H_c = P{k}'*Y1{k}+mu*P{k}'*X{k}-Y2{k}+Y2{k}*Z{k}'-mu*E{k}*Z{k}'+mu*E{k}-W1{k}+rho*Q{k};
        H{k} = lyap(H_a,H_b,H_c);       

        P_a = X{k} + Y1{k}/mu;
        P_b = H{k}*P_a';
        [svd_U,~,svd_V] = svd(P_b,'econ');
        P{k} = svd_V*svd_U';

        F = [H{1}-H{1}*Z{1}+Y2{1}/mu;H{2}-H{2}*Z{2}+Y2{2}/mu];
        [Econcat] = solve_l1l2(F,lambda2/mu);
        E{1} = Econcat(1:dim_H,:);
        E{2} = Econcat(dim_H+1:2*dim_H,:);

        Z_a = rho*eye(N) + mu*H{k}'*H{k};
        Z_b = rho*C{k} - W2{k} + H{k}'*Y2{k}+mu*H{k}'*(H{k}-E{k});
        Z{k} = Z_a\Z_b;

        Y1{k} = Y1{k} + mu*(X{k}-P{k}*H{k});
        Y2{k} = Y2{k} + mu*(H{k}-H{k}*Z{k}-E{k});
    end

    H_tensor = cat(3, H{:,:});
    W1_tensor = cat(3, W1{:,:});
    h = H_tensor(:);
    w1 = W1_tensor(:);
    [q, ~] = wshrinkObj(h + 1/rho*w1,1/rho,sX1,0,3);
    Q_tensor = reshape(q, sX1);

    Z_tensor = cat(3, Z{:,:});
    W2_tensor = cat(3, W2{:,:});
    z = Z_tensor(:);
    w2 = W2_tensor(:);
    [c, objV] = wshrinkObj(z + 1/rho*w2,lambda1/rho,sX2,0,3);
    C_tensor = reshape(c, sX2);

    w1 = w1 + rho*(h - q);
    w2 = w2 + rho*(z - c);

    history.objval(iter+1)   =  objV;

    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-P{k}*H{k},inf)>epson)
            history.norm_X = norm(X{k}-P{k}*H{k},inf);
            Isconverg = 0;
        end
        if (norm(H{k}-H{k}*Z{k}-E{k},inf)>epson)
            history.norm_H = norm(H{k}-H{k}*Z{k}-E{k},inf);
            Isconverg = 0;
        end
        
        Q{k} = Q_tensor(:,:,k);
        W1_tensor = reshape(w1, sX1);
        W1{k} = W1_tensor(:,:,k);
        C{k} = C_tensor(:,:,k);
        W2_tensor = reshape(w2, sX2);
        W2{k} = W2_tensor(:,:,k);
        if (norm(H{k}-Q{k},inf)>epson)
            history.norm_H_Q = norm(H{k}-Q{k},inf);
            Isconverg = 0;
        end
        if (norm(Z{k}-C{k},inf)>epson)
            history.norm_Z_C = norm(Z{k}-C{k},inf);
            Isconverg = 0;
        end
    end
   
    if (iter>200) 
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu = min(mu*pho_mu, max_mu);
    rho = min(rho*pho_rho, max_rho);
end
S = 0;
for k=1:K
    S = S + abs(Z{k})+abs(Z{k}');
end

NMI_total = zeros(1,30);
ACC_total = zeros(1,30);
F_total = zeros(1,30);
AVG_total = zeros(1,30);
Precision_total = zeros(1,30);
RI_total = zeros(1,30);
AR_total = zeros(1,30);
Recall_total = zeros(1,30);
for i_ave = 1:30
    C = SpectralClustering(S,cls_num);
    [A, nmi, avgent] = compute_nmi(gt,C);
    
    ACC = Accuracy(C,double(gt));
    [f,p,r] = compute_f(gt,C);
    [AR,RI,MI,HI]=RandIndex(gt,C);
    toc;
    fprintf('NMI: %f, ACC: %f, F-Score: %f\n',nmi,ACC,f);
    NMI_total(i_ave) = nmi;
    ACC_total(i_ave) = ACC;
    F_total(i_ave) = f;
    AVG_total(i_ave) = avgent;
    Precision_total(i_ave) = p;
    RI_total(i_ave) = RI;
    AR_total(i_ave) = AR;
    Recall_total(i_ave) = r;
end
NMI_mean = mean(NMI_total); NMI_std = std(NMI_total);
ACC_mean = mean(ACC_total); ACC_std = std(ACC_total);
F_mean = mean(F_total); F_std = std(F_total);
AVG_mean = mean(AVG_total); AVG_std = std(AVG_total);
Precision_mean = mean(Precision_total); Precision_std = std(Precision_total);
RI_mean = mean(RI_total); RI_std = std(RI_total);
AR_mean = mean(AR_total); AR_std = std(AR_total);
Recall_mean = mean(Recall_total); Recall_std = std(Recall_total);

fprintf('NMI: %f(%f), ACC: %f(%f), F-Score: %f(%f), Precision: %f(%f), AVG: %f(%f), RI: %f(%f), AR: %f(%f), Recall: %f(%f)\n',...
    NMI_mean,NMI_std,ACC_mean,ACC_std,F_mean,F_std,Precision_mean,Precision_std,AVG_mean,AVG_std,RI_mean,RI_std,AR_mean,AR_std,Recall_mean,Recall_std);
