% Load dataset
load FERETC80A45; % Each sample is a 32x32 matrix
[dc, dr, numSpl] = size(fea2D); % 32x32x320

% Partition the training and test sets
load DBpart; % 2 images per class for training, 2 image per class for test
fea2D_Train = fea2D(:, :, trainIdx);

% Heuristic 1D search for determining the regularization parameter \gamma
model = PROMA(fea2D_Train, 1, 'isReg', false, ...
    'maxIters', 100, 'tol', 1e-5);
gamma = model.sigma;

P = 500; % Reduce the dimensionality to 500
model = PROMA(fea2D_Train, P, 'isReg', true, ...
    'regParam', gamma, 'maxIters', 100, 'tol', 1e-5);

% PROMA Projection
newfea = projPROMA(fea2D, model);

% Sort the projected features by Fisher scores
[odrIdx, stFR] = sortProj(newfea(:,trainIdx), gnd(trainIdx));
newfea = newfea(odrIdx,:); 

% Classification via 1NN
dimTest = 200; % the number of features to be fed into a classifier
testfea = newfea(1:dimTest,:);
% In practice, it would be better to test different values of dimTest 
% for the best classification performance.

nnMd = fitcknn(testfea(:,trainIdx)', gnd(trainIdx));
label = predict(nnMd, testfea(:,testIdx)');
Acc = sum(gnd(testIdx) == label)/length(testIdx)
