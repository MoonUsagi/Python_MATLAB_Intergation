classdef kInputLayer20943 < nnet.layer.Layer & nnet.layer.Formattable
    %kInputLayer20943
    % Auto-generated by MATLAB on 25-Oct-2022 17:53:35
    
    
    properties
        % Non Trainable Parameters
        Variable
        Variable_1
    end
    
    properties (Learnable)
        % Trainable parameters
        
    end
    
    properties (Hidden)
        % Code literals
        
    end
    
    methods
        function obj = kInputLayer20943(Name, Type)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 1;
            obj.NumOutputs = 1;
        end
        
        function varargout = predict(obj, input)
            [temp{1}] = InputLayer20943(input, obj.Variable, obj.Variable_1, obj);
            
            % Extract results from function call.
            varargout{1} = temp{1}.value;
        end
    end
end

function [identity] = InputLayer20943(input, subReadvariableopResource, readvariableopResource, obj)
import DarkNet19_savedmodel.ops.*;
[input] = struct('value', input, 'rank', 4);

subReadVariableOp.value = subReadvariableopResource;
subReadVariableOp.rank = 3;
% Operation
[sub] = tfSub(input, subReadVariableOp);
ReadVariableOp.value = readvariableopResource;
ReadVariableOp.rank = 3;
sub1ReadVariableOp.value = subReadvariableopResource;
sub1ReadVariableOp.rank = 3;
% Operation
[sub1] = tfSub(ReadVariableOp, sub1ReadVariableOp);
% Operation
[truediv] = tfDiv(sub, sub1);
Identity.value = truediv.value;
Identity.rank = truediv.rank;


% assigning outputs
identity = Identity;
end
