function patch = get_patch(img, x, y, pr)
% Gets a patch of an image, with patch center (x,y) and patch radius pr

        %Print an error msg if patch center is too close to img border
        if(x < pr || y < pr || x > size(img,1) - pr || y > size(img,2) - pr)
    
            error('Patch outside image borders')%Print error
    
        else

        patch = img((x-pr):(x+pr),(y-pr):(y+pr),:); % create patch

        end
end