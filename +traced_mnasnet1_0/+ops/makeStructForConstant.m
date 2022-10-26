function tStruct = makeStructForConstant(data, rank, type)
%MAKESTRUCTFORCONSTANT Convert Constants to structs

%For tensor constants, data is dlarray labelled as 'U's
if isequal(type,"Tensor")
    if rank == 0
        label = 'U';
    else
        label =  repmat('U', [1, rank]);
    end
    data = dlarray(data,label);
elseif isequal(type,"Integer")
    if isa(data,"dlarray")
        data = int32(extractdata(data));
    else
        data = int32(data);
    end
end
tStruct = struct("value",data,"rank",rank);

end
