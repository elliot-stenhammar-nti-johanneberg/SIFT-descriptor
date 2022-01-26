
load digits.mat

% Creates a descriptor to all images in digits_training and adds it as a
% struct
for i = 1:1:length(digits_training)
    
    img = digits_training(i).image;
    cntr = ceil(size(img)/2);
    radius = (length(digits_training(i).image)-3)/6;
    
    digits_training(i).descriptor = gradient_descriptor(img,cntr,radius);
    
end