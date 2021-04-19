warning('off');
clear;clear memory;
nbits_set = [16,32,64,128];

iternum= 5;
mMAP = zeros(1,iternum);
nbits_map = zeros(1,length(nbits_set));
dataset_base = cell(4,1);
dataset_base{1} = '.\wikiData.mat';
dataset_base{2} = '.\mir_cnn.mat';
dataset_base{3} ='.\nus_cnn.mat';
dataset_base{4} = '.\coco2017_cnn.mat';

for i = 1:length(dataset_base)
    %% Load dataset
    dataset = dataset_base{i};
    load(dataset);
    if  strcmp(dataset, '.\wikiData.mat')
        X{1} = [I_tr;I_te];
        X{2} = [T_tr;T_te];
        gnd  = [L_tr;L_te];
    else
        X{1} = [I_db;I_te];
        X{2} = [T_db;T_te];
        gnd  = [L_db;L_te];
    end
    
    switch dataset
        case '.\wikiData.mat'
            beta = .5;
            n_anchor =500;
            pars.lambda = 10^(-3);
            pars.theta = 1;
        case '.\nus_cnn.mat'
            beta = .5;
            n_anchor = 1000;
            pars.lambda = 10^(-1);
            pars.theta = 1;
        case '.\coco2017_cnn.mat'
            beta = .5;
            n_anchor = 1000;
            pars.lambda = 10^(-1);
            pars.theta = 10;
        case '.\mir_cnn.mat'
            beta = .5;
            n_anchor = 1000;
            pars.lambda = 10^(-7);
            pars.theta = 10;
    end
    
    %% Anchor feature embedding
    view_num = size(X,2);
    Anchor = cell(1,view_num);
    n_Sam = size(X{1},1);
    Wour= cell(1,view_num);
    for it = 1:view_num
        X{it} = double(X{it});
        anchor = X{it}(randsample(n_Sam, n_anchor),:);
        Dis = EuDist2(X{it},anchor,0);
        sigma = mean(mean(Dis)).^0.5;
        feavec = exp(-Dis/(2*sigma*sigma));
        X{it} = bsxfun(@minus, feavec', mean(feavec',2));
    end
    
    %% Separate Train and Test Index
    tt_num = size(I_te,1);
    data_our.gnd = gnd;
    tt_idx = n_Sam-tt_num+1:n_Sam;
    tr_idx = 1:n_Sam-tt_num;
    ttgnd = gnd(tt_idx,:);
    trgnd = gnd(tr_idx,:);
    clear gnd;
    pars.beta = beta;
    %% fixed parameters
    pars.gamma_SPL    = 100;
    pars.Iter_num = 25;
    pars.L21epsilon = 10;
    pars.deep_lambda1       =10;
    pars.deep_lambda2       =10;
    pars.deep_eps1          =.5;
    pars.deep_eps2          =.5;
    pars.deep_mu            = 0.01;
    
    %% different hash length
    for ii=1:length(nbits_set)
        nbits = nbits_set(ii);
        pars.nbits    = nbits;
        pars.deep_numOfAtoms1   = 120;
        data_our.indexTrain= tr_idx;
        data_our.indexTest= tt_idx;
        ttfea = cell(1,view_num);
        for view = 1:view_num
            data_our.X{view} = normEqualVariance(X{view}')';
            ttfea{view} = data_our.X{view}(:,tt_idx);
        end
        
        switch dataset
            case '.\wikiData.mat'
                iter4w = 20;
                iter4r = 1;
            case  '.\mir_cnn.mat'
                iter4w =25;
                iter4r = 25;
            case  '.\nus_cnn.mat'
                iter4w = 15;
                iter4r = 25;
            case '.\coco2017_cnn.mat'
                iter4w = 5;
                iter4r = 5;
        end
        
        for n_iters = 1:iternum
            pars.n_iters = n_iters;
            [B_trn, W, U_W, R, P, alpha, trtime] = CMCH_fun(data_our,pars,iter4w,iter4r);
            %% for testing
            V = zeros(nbits_set(ii),tt_num);
            for ind = 1:size(ttfea,2)
                V = V+alpha(ind)*W{ind}*ttfea{ind};
            end
            % generate self-adaptative alpha for test samples
            Ialpha = ones(1,view_num);
            
            VT = V';
            for ind = 1:view_num
                Wour{ind}=W{ind}*data_our.X{ind}(:, data_our.indexTest);
            end
            
            Hquad = zeros(view_num,view_num);
            for i = 1:view_num
                for j = 1:view_num
                    Hquad (i,j) =  trace( Wour{i}* Wour{j}' );
                end
            end
            
            fquad = zeros(view_num,1);
            for i = 1:view_num
                fquad (i) =  trace( Wour{i}* VT );
            end
            ub = ones(view_num,1);
            lb = zeros(view_num,1);
            alpha = quadprog(Hquad,-fquad,[],[],Ialpha,1,lb,ub);
            for ind = 1:size(ttfea,2)
                V = V+alpha(ind)*W{ind}*ttfea{ind};
            end
            %% generate test hash codes
            B_tst = V'*U_W >0;
            %% Groungdtruth
            if  strcmp(dataset, '.\wikiData.mat')
                WtrueTestTraining = bsxfun(@eq, ttgnd, trgnd');
            else
                WtrueTestTraining = ttgnd * trgnd'>0;
            end
            %% Evaluation
            B1 = compactbit(B_trn);
            B2 = compactbit(B_tst);
            DHamm = hammingDist(B2, B1);
            [~, orderH] = sort(DHamm, 2);
            MAP = calcMAP(orderH, WtrueTestTraining);
            mMAP(n_iters)=MAP;
            fprintf('Processing %s\n ',dataset);
            fprintf(' The %d-th test of %d bits hashing codes, Map: %.4f...   \n', n_iters, nbits, MAP);
        end
        nbits_map(ii) = mean(mMAP(:));
    end
    disp(nbits_map);
    fprintf('%s finished\n',dataset);
end

fprintf('finish');

