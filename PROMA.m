function model = PROMA( TX, P, varargin )
% PROMA: Probabilistic Rank-One Matrix Analysis
%
% %[Syntax]%: 
%   model = PROMA( TX, P )
%   model = PROMA( ___, Name, Value )
%
% %[Inputs]%:
%   TX:            the dc x dr x numSpl training set of input matrices 
%   P:             the number of extraced features
%     
% %[Name-Value Pairs]
%   'isReg':       the indicator for performing concurrent regularization
%                  true (Default) | false
%
%   'regParam':    the regularization parameter \gamma
%                  1e3 (Default)
%
%   'maxIters':    the maximum number of iterations
%                  100 (Default)
%
%   'tol':         the tolerance of the relative change of log-likelihood
%                  1e-5 (Default)
%
% %[Outputs]%:
%   model.C:       the column factor matrix
%   model.R:       the row factor matrix
%   model.sigma:   the noise variance
%   model.TXmean:  the mean of the training matrices TX
%   model.liklhd:  the log-likelihood at each iteration
%                        
% %[Toolbox needed]%:
%   This function needs the tensor toolbox v2.6 available at
%   http://www.sandia.gov/~tgkolda/TensorToolbox/
%                       
% %[Reference]%:            
%   Yang Zhou, Haiping Lu. 
%   Probabilistic Rank-One Matrix Analysis with Concurrent Regularization. 
%   in Proceedings of IJCAI 2016, New York, USA, 
%   pp. 2428-2434, July 09-15, 2016.
%                             
% %[Author Notes]%   
%   Author:        Yang ZHOU
%   Email :        yangzhou@comp.hkbu.edu.hk
%   Affiliation:   Department of Computer Science
%                  Hong Kong Baptist University
%   Release date:  Sept. 29, 2017
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 

if nargin < 2, error('Not enough input arguments.'); end
[dc, dr, numSpl] = size(TX);

ip = inputParser;
ip.addParameter('isReg', true, @islogical);
ip.addParameter('regParam', 1e3, @isscalar);
ip.addParameter('maxIters', 100, @isscalar);
ip.addParameter('tol', 1e-5, @isscalar);
ip.parse(varargin{:});

isReg  = ip.Results.isReg;
gamma = ip.Results.regParam;
maxK = ip.Results.maxIters; 
epsCvg = ip.Results.tol; 

%   Random initialization
C = rand(dc, P); R = rand(dr, P); 
C = normc(C); R = normc(R);
if isReg ~= 1 % W/o concurrent regularization
    % Noise variance is initialized randomly
    sigma = rand(1);
else % With concurrent regularization
    % Noise variance is fixed as \gamma
    sigma = gamma;
end

%   Data Centering
TXmean = mean(TX,3); % The mean matrix 
TX = bsxfun(@minus,TX,TXmean);
X_vec = reshape(TX, dc*dr, numSpl); % Vectorized dataset

%   PROMA Iterations
liklhd = zeros(maxK,1);
for iI = 1 : maxK    
%   Calc Expectation    
    CC = C'*C; RR = R'*R;
    Minv = eye(P)/(CC.*RR + sigma*eye(P));
    
    W = khatrirao(R,C); 
    expZ = Minv * W' * X_vec;
    covZ = numSpl*sigma*Minv + expZ*expZ';
    
%   Update C
    C = zeros(dc,P);
    for m = 1:numSpl
        Xm = TX(:,:,m);
        C = C + Xm*R*diag(expZ(:,m));
    end
    C = C / (covZ.*(R'*R));
    
%   Update R
    R = zeros(dr,P);
    for m = 1:numSpl
        Xm = TX(:,:,m);
        R = R + Xm'*C*diag(expZ(:,m));
    end
    R = R / (covZ.*(C'*C));

    if isReg ~= 1 %	W/o concurrent regularization    
        %   Update sigma            
        trXX = sum(sum(X_vec.^2));
        trCZRCZR = sum(sum((C'*C).*(R'*R).*covZ));
        sigma = (trXX - trCZRCZR) / (numSpl*dc*dr);
    end
        
%   Calc Likelihood
    W = khatrirao(R,C); 
    G = W*W' + sigma*eye(dc*dr); % Total covariance matrix
    
%   Exact Likelihood (Maybe slow with hihg-dimensional inputs)
    eigG = eig(G);
    det = sum(log(eigG)); % Calc log determinant
    nloglk = - 0.5*(numSpl*dc*dr*log(2*pi) + numSpl*det + sum(sum(X_vec'/G.*X_vec',2))); 
    
%   Approximated Likelihood
%     nloglk = -0.5*((P+dc*dr)*log(2*pi)+numSpl*dc*dr*(log(sigma)+1)+sum(sum(expZ.^2)));

    % Check Convergence:
    liklhd(iI) = nloglk;
    if iI > 1
        threshold = abs(liklhd(iI) - liklhd(iI-1)) / abs(liklhd(iI));
        if threshold < epsCvg, 
            disp('Log Likelihood Converge.'); 
            liklhd = liklhd(1:iI);
            break; 
        end
        fprintf('Iteration %u, Likelihood = %f, Threshold = %f.\n', iI, liklhd(iI), threshold);
    else 
        fprintf('Iteration %u, Likelihood = %f.\n', iI, liklhd(1));
    end
end

model.C = C; model.R = R;
model.sigma = sigma;
model.TXmean = TXmean;
model.liklhd = liklhd;
end

