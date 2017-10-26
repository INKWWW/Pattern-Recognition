%% =======Load  MINST label======= %%

function labels = loadlabel(filename)  
%load MNIST Labels and returns a [number of MNIST images]x1 matrix containing  
%the labels for the MNIST images
  
    fp = fopen(filename, 'rb'); 
    assert(fp ~= -1, ['Could not open ', filename, '']);
    
    %read the fisrt 2 rows' description information (big-endian model)
    magic = fread(fp, 1, 'int32', 0, 'b');  
    assert(magic == 2049, ['Bad magic number in ', filename, '']);
    numLabels = fread(fp, 1, 'int32', 0, 'b'); 

    %read labels according to the offset value until end
    labels = fread(fp, inf, 'unsigned char'); 
    assert(size(labels,1) == numLabels, 'Mismatch in label count');  

    %close file
    fclose(fp);
  
end  