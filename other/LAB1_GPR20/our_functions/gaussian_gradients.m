function [grad_x,grad_y] = gaussian_gradients(img,std)
% Inputs a grayscale image + std and outputs 2 matrices with pixel
% derivatives in x and y directions.

    Ix = [-0.5 0 0.5]; %Derivative filter in x direction
    Iy = [-0.5;0;0.5]; %Derivative filter in y direction
    
    h = gaussian_filter(img,std); %Filter img
   
    grad_x = imfilter(h,Ix,'symmetric'); %Creates grad_x 
    grad_y = imfilter(h,Iy,'symmetric'); %Creates grad_y 

end