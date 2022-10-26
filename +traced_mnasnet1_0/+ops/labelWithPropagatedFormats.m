function [dlXOut] = labelWithPropagatedFormats(dlX,formats)
% Function to determine the label a dlarray with the propagated format.
% The U-labelled dlarray is assumed to be in reverse python format. Permute
% to DLT format and label. 
% Since batch is optional in PyTorch, we use rank with propagated label to 
% determine the format and permutations 

%   Copyright 2022 The MathWorks, Inc.

dlXRank = dlX.rank;
if all(dims(dlX.value) =='U')
    formatWithRank = strcat(formats,string(dlXRank));
    switch formatWithRank
        case '*CSS3'
            permReversePythonToDLT       = [2 1 3];
            outputFormat                 = 'SSC';
        case '*CSS4'
            permReversePythonToDLT       = [2 1 3 4];
            outputFormat                 = 'SSCB';
        case '*CSSS4'
            permReversePythonToDLT       = [3 2 1 4];
            outputFormat                 = 'SSSC';
        case '*CSSS5'
            permReversePythonToDLT       = [3 2 1 4 5];
            outputFormat                 = 'SSSCB';
        case '*C2'
            permReversePythonToDLT       = [1 2];
            outputFormat                 = 'CB';
        otherwise
            error('Unknown format and rank %s', formatWithRank);
    end
    
    dlXValue    = dlX.value;
    dlXPermuted = permute(stripdims(dlXValue),permReversePythonToDLT);
    dlXOut.value = dlarray(dlXPermuted,outputFormat);
    dlXOut.rank = dlXRank;
else
    dlXOut = dlX;
end

end