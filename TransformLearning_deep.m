function [P1, P2] = TransformLearning_deep (P1, Y, lambda, deep_par, B )
lambda1=deep_par.lambda1;
lambda2=deep_par.lambda2;
eps1=deep_par.eps1;
eps2=deep_par.eps2;
type = 'hard';
invL1 = (Y*Y' + lambda1*eps1*eye(size(Y,1)))^(-0.5);

%     update Coefficient Z sparse
switch type
    case 'soft'
        Z1 = sign(P1*Y).*max(0,abs(P1*Y)-lambda);
    case 'hard'
        Z1 = (abs(P1*Y) >= lambda) .* (P1*Y);
end
[U,S,V] = svd(invL1*Y*Z1');
D1 = [diag(diag(S) + (diag(S).^2 + 2*lambda1).^0.5) ];
DS = blkdiag(D1,zeros(size(V,2)-size(D1,1),size(U,2)-size(D1,2)));
P1 = 0.5*V*DS*U'*invL1;
Z2 = B;
invL2 = (Z1*Z1' + lambda2*eps2*eye(size(Z1,1)))^(-0.5);
[U,S,V] = svd(invL2*Z1*Z2');
D2 = [diag(diag(S) + (diag(S).^2 + 2*lambda2).^0.5) ];
DS = blkdiag(D2,zeros(size(V,2)-size(D2,1),size(U,2)-size(D2,2)));
P2 = 0.5*V*DS*U'*invL2;

end
