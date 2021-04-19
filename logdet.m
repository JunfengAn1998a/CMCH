function y = logdet(A)

try
    U = chol(A);
    y = 2*sum(log(diag(U))) ;
catch 
    y = 0;
    warning('logdet:postdef', 'Matrix is not positive definite');
end

end