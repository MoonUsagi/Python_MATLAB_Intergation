function [dlXPerm, permReversePythonToDLT] = permuteToReversePyTorch(dlX)
% Function to determine the permutation vectors from DLT format to
% reverse-Python and vice-versa, for a given labeled dlarray dlX. Permutes 
% dlX into reverse-Python dimension ordering, returning the result as dlXPerm.
% The result will be unlabeled.

%   Copyright 2022 The MathWorks, Inc.

if isdlarray(dlX) && ~isempty(dims(dlX)) && ~any(dims(dlX) == 'U')
    labels = dims(dlX);    
    switch labels
        case 'SSCB'
            permDLTToReversePython = [2 1 3 4]; % HWCN -> WHCN
            permReversePythonToDLT = [2 1 3 4]; % WHCN -> HWCN
        case 'SSC'
            permDLTToReversePython = [2 1 3]; % HWC -> WHC
            permReversePythonToDLT = [2 1 3]; % WHC -> HWC         
        case 'SSSCB'
            % NOTE: Although fwd-PyTorch canonically uses the format "NCDHW",
            % we choose to preserve the order of the spatial dimensions,
            % treating them as HWD rather than DHW.
            permDLTToReversePython = [3 2 1 4 5]; % HWDCN -> DWHCN
            permReversePythonToDLT = [3 2 1 4 5]; % DWHCN -> HWDCN        
        case 'SSSC'
            permDLTToReversePython = [3 2 1 4]; % HWDC -> DWHC
            permReversePythonToDLT = [3 2 1 4]; % DWHC -> HWDC            
        case 'CB'
            permDLTToReversePython = [1 2]; % CN -> CN
            permReversePythonToDLT = [1 2]; % CN -> CN           
        otherwise 
            error(message('nnet_cnn_pytorchconverter:pytorchconverter:DlarrayFormatNotRecognized', labels));
    end
    dlXPerm = permute(stripdims(dlX), permDLTToReversePython);
else
    % dlX is already in reverse-Python dimension order. Return it
    % unchanged, and leave permReversePythonToDLT empty.
    permReversePythonToDLT = [];    
    dlXPerm = stripdims(dlX);
end

end