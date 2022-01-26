function result = gaussian_filter(img, std)
%This function takes an image and a standard-deviation as inputs
%And outputs a filtered image

    h = fspecial('gaussian',ceil(4*std+1),std); %Create filter (gaussian)
    result = imfilter(img,h,'symmetric'); %Filter image with h
    
end