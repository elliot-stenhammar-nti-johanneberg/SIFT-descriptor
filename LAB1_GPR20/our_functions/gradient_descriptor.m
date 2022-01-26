function desc = gradient_descriptor(img, pos, radius)
    
    std = radius/4; % standard deviation = radius*k as in desqription
    centr= place_regions(pos,radius); % gives 2 by 9 vector with centers
    desc = zeros(72,1); % empty descriptor
    
    [grad_x,grad_y] = gaussian_gradients(img,std); % creates gradiants for the img
    
    % This for loop creates a histogram for each patch and puts it into the
    % corresponding place in the descriptor 
    for i = 1:1:9
        patch_x = get_patch(grad_x, centr(1,i), centr(2,i), radius);
        patch_y = get_patch(grad_y, centr(1,i), centr(2,i), radius);
        histogram = gradient_histogram(patch_x,patch_y);
        desc(8*(i-1)+1:8*i)=histogram;
    end
    desc=desc/norm(desc); % normalizes the descriptor before returning it
end 