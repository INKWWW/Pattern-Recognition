%binarize the features
function [xt] = binarization(x)
    s = size(x);
    xt = zeros(s);
    for m = 1:s(1,1)
        for n = 1:s(1,2)
            if x(m,n)==0
                xt(m,n)=0;
            else
                xt(m,n)=1;
            end
        end
    end
end
            