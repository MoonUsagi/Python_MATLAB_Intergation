classdef aten__linear12 < nnet.layer.Layer & nnet.layer.Formattable
    %aten__linear12 Auto-generated custom layer
    % Auto-generated by MATLAB on 26-Oct-2022 11:08:24
    
    properties (Learnable)
        % Networks (type dlnetwork)
        
    end
    
    properties
        % Non-Trainable Parameters
        
    end
    
    properties (Learnable)
        % Trainable Parameters
        
        TopLevelModule_classifier_1_weight
        TopLevelModule_classifier_1_bias
    end
    
    methods
        function obj = aten__linear12(Name, Type, InputNames, OutputNames)
            obj.Name = Name;
            obj.Type = Type;
            obj.NumInputs = 2;
            obj.NumOutputs = 1;
            obj.InputNames = InputNames;
            obj.OutputNames = OutputNames;
        end
        
        function [linear_9] = predict(obj,linear_argument1_1, linear_argument1_1_rank)
            
            if ~contains(dims(linear_argument1_1),'U')
                [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', ndims(linear_argument1_1));
            else
                [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', int32(numel(linear_argument1_1_rank)));
            end
            
            import traced_mnasnet1_0.ops.*;
            
            linear_weight_1 = obj.TopLevelModule_classifier_1_weight;
            
            [linear_weight_1] = struct('value', linear_weight_1, 'rank', 2);
            
            linear_bias_1 = obj.TopLevelModule_classifier_1_bias;
            
            [linear_bias_1] = struct('value', linear_bias_1, 'rank', 1);
            
            [linear_9] = pyLinear(linear_argument1_1, linear_weight_1, linear_bias_1);
            
            linear_9 = linear_9.value ;
            
        end
        
        
        
        function [linear_9] = forward(obj,linear_argument1_1, linear_argument1_1_rank)
            
            if ~contains(dims(linear_argument1_1),'U')
                [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', ndims(linear_argument1_1));
            else
                [linear_argument1_1] = struct('value', linear_argument1_1, 'rank', int32(numel(linear_argument1_1_rank)));
            end
            
            import traced_mnasnet1_0.ops.*;
            
            linear_weight_1 = obj.TopLevelModule_classifier_1_weight;
            
            [linear_weight_1] = struct('value', linear_weight_1, 'rank', 2);
            
            linear_bias_1 = obj.TopLevelModule_classifier_1_bias;
            
            [linear_bias_1] = struct('value', linear_bias_1, 'rank', 1);
            
            [linear_9] = pyLinear(linear_argument1_1, linear_weight_1, linear_bias_1);
            
            linear_9 = linear_9.value ;
            
        end
        
        
    end
end
