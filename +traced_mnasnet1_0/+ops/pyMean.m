function Y = pyMean(X, dim, keepdim)
%PYMEAN Calculates the mean of the input tensor along dimension dim.

import traced_mnasnet1_0.ops.*

dim = [dim.value];
keepdim = keepdim.value;

% If dim is empty, get all dims
if isempty(dim)
    dim = 0:X.rank-1; % 0-indexed, as in PyTorch
end

% Convert data and dims to reverse-PyTorch dimension ordering
[Xval, permToDLT] = permuteToReversePyTorch(X.value);
mlDims = X.rank - dim;

% Compute the mean
Yval = mean(Xval, mlDims);

% If keepdim is false (the default), squeeze out dims
if ~keepdim || isempty(keepdim)
    % Get size vector (including singletons up to rank)
    sz = ones(1, X.rank);
    ySz = size(Yval);
    sz(1:numel(ySz)) = ySz;
    % Remove dims that were averaged across
    for i=1:numel(mlDims)
        sz(mlDims(i)) = [];
    end
    % If sz has less than 2 elements, append trailing singletons
    if numel(sz) < 2
        sz = [sz ones(1, 2-numel(sz))];
    end
    % Reshape to remove dims
    Yrank = X.rank - numel(dim);
    Yval = dlarray(reshape(Yval, sz));   
    % Set output format if input was SSCB and SS dimensions were reduced.
    inputFmt = char(dims(X.value));
    keptDims = setdiff(1:X.rank, mlDims);
    if inputFmt=="SSCB" && inputFmt(keptDims)=="CB"
        Yval = dlarray(reshape(Yval, sz), 'CB');   
    else
        Yval = dlarray(reshape(Yval, sz), repmat('U', 1, max(Yrank, 2)));   
    end
else
    % Permute back and reassign labels if possible
    if ~isempty(permToDLT)
        Yval = permute(Yval, permToDLT);
    end
    Yval = dlarray(Yval, dims(X.value));
    Yrank = X.rank;    
end
Y = struct('value', Yval, 'rank',Yrank);
end