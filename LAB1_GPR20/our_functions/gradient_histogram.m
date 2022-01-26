function histogram = gradient_histogram(grad_x,grad_y)
% This function calculates the magnitude and angle of all pixel gradiant
% and puts them into corresponding histogram bin

histogram = zeros(8,1); % Creates an empty  histogram
for i = 1:1:size(grad_x,1)*size(grad_x,2)
   
    mag = sqrt((grad_x(i)^2 + grad_y(i)^2)/2); % mag. for each pixel
    ang = mod(atan2(-grad_y(i),grad_x(i))*(180/pi),360); % Angle in degrees
    
    % This for loop puts the current mag into the correct bin
    for j = 1:1:8
        if( ang < 360-45*(j-1) && ang > 360-45*j)
            histogram(j) = histogram(j)+mag;
        end
    end
end
end