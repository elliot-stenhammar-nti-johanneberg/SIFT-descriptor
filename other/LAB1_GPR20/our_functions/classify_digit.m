function label = classify_digit(digit_img,digits_training)
% This function input a gratscale image and a training struc which must
% contain descriptors and lables. It outputs the lable of what
% digit_image is classified as

% Creates a descriptor for the image to classify
img = digit_img;
cntr = ceil(size(img)/2);
radius = (length(img)-3)/6;
img_desc = gradient_descriptor(img,cntr,radius);

best_match = 0; %init holder for i of best match
min_sum = 1000; %Init holder for the smallest sum

% for loop that runs the length of digits training
for i =  1:1:length(digits_training)
        
    %compute the diffrence of descriptors 
    
    % In this case this method of calculating the sum yeilds a better
    % result
    curr_sum = sum(abs(img_desc - digits_training(i).descriptor));
    
    % This method was more common in practice
    %curr = sqrt(sum((img_desc - digits_training(i).descriptor).^2));
      
    if curr_sum < min_sum
        
        min_sum = curr_sum; % Saves the smallest sum
        best_match = i; % Saves i for  best match
        
    end
end
% disp(['its a ' num2str(digits_training(best_match).label)])
label=digits_training(best_match).label;
end