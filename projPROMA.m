function newfea = projPROMA( TX, model)
% PROMA projection with the trained model
%
% %[Syntax]%: 
%    newfea = projPROMA( TX, model)
%
% %[Inputs]%:
%    TX:            the dc x dr x numSpl input matrix set
%    model:         the trained PROMA model
%
% %[Outputs]%:
%    newfea:        the projected P-dimensional features
%
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/

    C = model.C; R = model.R;    
    sigma = model.sigma;
    Mu = model.TXmean;
    
    P = size(C,2);
    [dc, dr, numSpl] = size(TX);
    
    TX = bsxfun(@minus,TX, Mu); %Centering
    X_vec = reshape(TX, dc*dr, numSpl); % Vectorization
    
    % Compute M
    CC = C'*C; RR = R'*R;
    Minv = eye(P)/(CC.*RR + sigma*eye(P));
    
    % Projection
    W = khatrirao(R,C); 
    newfea = Minv * W' * X_vec;
end

