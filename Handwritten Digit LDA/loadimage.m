%% ========Load MINST images======== %%

function images = loadimage(filename)  
%load MNIST Images and returns a 28x28x[number of MNIST images] matrix containing  
%the raw MNIST images  
  
    fp = fopen(filename, 'rb');   %read as binary 
    assert(fp ~= -1, ['Could not open ', filename, '']);  %if fopen fails

    %read the fisrt 4 rows' description information (big-endian model)
    magic = fread(fp, 1, 'int32', 0, 'b');
    assert(magic == 2051, ['Bad magic number in ', filename, '']);
    
    numImages = fread(fp, 1, 'int32', 0, 'b');  
    numRows = fread(fp, 1, 'int32', 0, 'b');  
    numCols = fread(fp, 1, 'int32', 0, 'b');
    
    %read the images according to the offset value
    images = fread(fp, inf, 'unsigned char'); %read unsigned char=8 bits once until end
    images = reshape(images, numCols, numRows, numImages); %reshape matrix to numCols*numRows*numImages
    images = permute(images,[2 1 3]); %exchange the numCols*numRows to numRows*numCols

    %close file
    fclose(fp); 

    % Reshape matrix to (num of pixels) * (num of images)  
    images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));  
    % Convert to double and regularize to [0,1] 
    images = double(images) / 255; %same as im2double coz images is niut8 type
  
end  