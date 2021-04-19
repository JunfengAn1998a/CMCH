function [B_train, W, U_W, R, P, alpha, trtime] = CMCH_fun(data, pars,iter4w,iter4r)
n_iters = pars.n_iters;
deeppars.numOfAtoms1 = pars.deep_numOfAtoms1;
deeppars.lambda1 = pars.deep_lambda1;
deeppars.lambda2 = pars.deep_lambda2;
deeppars.eps1 = pars.deep_eps1;
deeppars.eps2 = pars.deep_eps2;
deeppars.mu = pars.deep_mu;
gamma_SPL     = pars.gamma_SPL;
beta     = pars.beta;
lambda   = pars.lambda;
theta = pars.theta;
lambdadivtheta = lambda/theta;
Iter_num = pars.Iter_num;
nbits    = pars.nbits;
trIdx = data.indexTrain;
L21epsilon=pars.L21epsilon;
view_num = size(data.X,2);
XXT = cell(1,view_num);
XT = cell(1,view_num);
dim = zeros(1,view_num);
our_data = cell(1,view_num);
Wour= cell(1,view_num);
WourTimesRspl= cell(1,view_num);
for ind = 1:view_num
    our_data{ind} = data.X{ind}(:,trIdx);
    [dim(ind), n] = size(our_data{ind});
    XXT{ind} = our_data{ind}*our_data{ind}';
end

for ind = 1:view_num
    XT{ind} = our_data{ind}';
end
SPLXT =  cell(1,view_num);
for ind = 1:view_num
    SPLXT{ind} = our_data{ind}';
end
betavec =beta*ones(size(our_data{1},2),1);
if isvector(data.gnd)
    L_tr = data.gnd(trIdx);
    Y = sparse(1:length(L_tr), double(L_tr), 1); Y = full(Y');
else
    L_tr = data.gnd(trIdx,:);
    Y = double(L_tr');
end

% %%%%%% B - initialize %%%%%%
alpha = ones(view_num,1)/view_num;
B = randn(nbits, n)>0; B = B*2-1;
P = randn(nbits, size(Y,1));
W = cell(1,view_num);
ub = ones(view_num,1);
lb = zeros(view_num,1);
Ialpha = ones(1,view_num);
for ind = 1:view_num
    W{ind} = randn(nbits, dim(ind));
end
V = P*Y;

[U0,~,P0] = svd(V*B', 'econ');
R = U0*P0';

WX = zeros(nbits,n);
for ind = 1:size(our_data,2)
    WX = WX+alpha(ind)*W{ind}*our_data{ind};
end

LOSS = zeros(size(V));
L_SPL = zeros(size(LOSS,2),1);
R_SPL = ones(size(LOSS,2),1);
W21_tempitem = cell(1,view_num);
for ind = 1:view_num
    W21_tempitem{ind} = randn(size(W{ind},2), 1);
end
for ind = 1:view_num
    for colum = 1:size(W{ind},2)
        W21_tempitem{ind}(colum) = 1/sqrt(norm(W{ind}(:,colum))^2+L21epsilon);
    end
end
P1 = randn(deeppars.numOfAtoms1, size(Y,1));
tic;

%------------------------training----------------------------
for iter = 1:Iter_num
    fprintf('The %d-th train of %d bits hashing. The %d-th iteration...\n',n_iters,nbits,iter);
    % ----------------------- W-step -----------------------%
    if iter<=iter4w
        for ind = 1:view_num
            for colum = 1:size(W{ind},2)
                W21_tempitem{ind}(colum) = 1/sqrt(norm(W{ind}(:,colum))^2+L21epsilon);
            end
        end
        WX = zeros(nbits,n);
        for ind = 1:size(our_data,2)%M
            Con = zeros(size(V));
            
            for j = 1:size(our_data,2)%M
                if j==ind
                    continue;
                else
                    Con = Con + alpha(j)*eye(size(W{j},1))*W{j}*our_data{j};
                end
            end
            Con = Con - V;
            SPLXT{ind}=bsxfun(@times,R_SPL,XT{ind});
            W{ind} = -1*alpha(ind)*eye(size(V,1))*Con*our_data{ind}'/(alpha(ind)*alpha(ind)*eye(size(XXT{ind},1))*our_data{ind}*SPLXT{ind}+2*lambda*eye(size((alpha(ind)*alpha(ind)*eye(size(XXT{ind},1))*our_data{ind}*SPLXT{ind}))));
        end
    end
    % ----------------------- R-step -----------------------%
    if iter <= iter4r
        [U0,~,P0] = svd(V*B', 'econ');
        R = U0*P0';
    end
    % ----------------------- P-step -----------------------%
    [P1, P2] = TransformLearning_deep (P1, Y, lambdadivtheta, deeppars, B);
    % ----------------------- V-step -----------------------%
    WS = zeros(nbits,size(our_data{1},2));
    for ind = 1:view_num
        WS = WS + alpha(ind)*W{ind}*our_data{ind};
    end
    
    WSSPL = zeros(size(WS));
    for colum = 1:size(WS,2)
        WSSPL(:,colum) = R_SPL(colum)*WS(:,colum);
    end
    me = (WSSPL + beta*R'*B);
    V = me;
    Vde = betavec+R_SPL;
    Vde = 1./Vde;
    
    for colum = 1:size(V,2)
        V(:,colum) = Vde(colum)*V(:,colum);
    end
    
    % ----------------------- B-step -----------------------%
    Z1 = (abs(P1*Y) >= lambdadivtheta) .* (P1*Y);
    PY = (abs(P2*Z1) >= lambdadivtheta) .* (P2*Z1);
    A = beta*R*V+theta*PY;
    mu = median(A,2);
    A = bsxfun(@minus, A, mu);
    B = sign(A);
    
    % ----------------------- alpha-step -----------------------%
    if iter>0
        VT = V';
        for ind = 1:view_num
            Wour{ind}=W{ind}*our_data{ind};
            for column = 1:size(Wour{ind},2)
                WourTimesRspl{ind}(:,column) = Wour{ind}(:,column) *R_SPL(column);
            end
        end
        
        Hquad = zeros(view_num,view_num);
        for i = 1:view_num
            for j = 1:view_num
                Hquad (i,j) =  trace( WourTimesRspl{i}* Wour{j}' );
            end
        end
        
        fquad = zeros(view_num,1);
        for i = 1:view_num
            fquad (i) =  trace( WourTimesRspl{i}* VT );
        end
        
        alpha = quadprog(Hquad,-fquad,[],[],Ialpha,1,lb,ub);
        alpha = round(100*alpha) /100;
        
        WX = zeros(nbits,n);
        for ind = 1:size(our_data,2)
            WX = WX+alpha(ind)*W{ind}*our_data{ind};
        end
    end
    
    % ----------------------- R_SPL-step -----------------------%
    LOSS = zeros(size(V));
    for ind = 1:view_num
        LOSS = LOSS + alpha(ind)*W{ind}*our_data{ind};
    end
    LOSS = LOSS - V;
    LOSS = LOSS';
    
    for ind = 1:size(LOSS,2)
        L_SPL(ind) = norm(LOSS(:,ind),'fro');
    end
    
    L_SPL = L_SPL';
    L_SPL = mapminmax(L_SPL, 0, 1);
    L_SPL = L_SPL';
    gamma_SPL = 1.1 * gamma_SPL;
    
    for ind = 1:size(L_SPL,1)
        me = (1+exp(-1*gamma_SPL));
        de = (1+exp(L_SPL(ind)-gamma_SPL));
        R_SPL(ind) = me/de;
    end
    
end
trtime = toc;
B_train = B'>0;

% Out-of-Sample
V0 = zeros(nbits,n);
for ind = 1:view_num
    V0 = V0+alpha(ind)*W{ind}*our_data{ind};
end
NT = (V0*V0' + 1 * eye(size(V0,1))) \ V0;
U_W = NT*B_train;

end