function Y = pyLinear(X, W, B)
%PYLINEAR Applies a linear transformation to the input data.
% at::Tensor at::linear(const at::Tensor &input, const at::Tensor &weight, const c10::optional<at::Tensor> &bias = {})

import traced_mnasnet1_0.ops.*

% Convert the input data to reverse-Python dimension order
[Xval, permXToDLT] = permuteToReversePyTorch(X.value);
Wval = permuteToReversePyTorch(W.value);
if ~isempty(B.value)
    Bval = permuteToReversePyTorch(B.value);
else
    Bval = B.value;
end
Xrank = X.rank;
Yrank = Xrank;

% If X is a vector, ensure it is a column vector
if Xrank==1
    Xval = [Xval(:)];
end

% If B is a vector, ensure it is a column vector
if B.rank ==1
    Bval = [Bval(:)];
end

% The PyTorch input format for X
% (*, H_in), where * can be any number of dimensions, and H_in is the
% number of input features. W has format (H_out, H_in) and is transposed
% before being multipled by X. 
% In reverse Python, a 4-dimensional BCSS input will have format 
% (H_in, x3, x1, x2). 
% W will have shape (H_in, H_out). To multiply
% equivalent pages of X and W, pages of X must be transposed. 
Yval = pagemtimes(Xval, 'transpose', Wval, 'none');

% After multiplication, pages of Yval are in reverse dimension order.
perm = 1:Yrank;
perm(2) = 1;
perm(1) = 2;
Yval = permute(Yval, perm);

% Add bias 
if ~isempty(Bval)
    Yval = Yval + Bval;
end

% Permute back to original dimension order and reapply labels
if ~isempty(permXToDLT)
    Yval = permute(Yval, permXToDLT);
    Yval = dlarray(Yval, dims(X.value));
else
    Yval = dlarray(Yval, repmat('U', 1, Yrank));
end
Y = struct('value', Yval, 'rank', Xrank);
end