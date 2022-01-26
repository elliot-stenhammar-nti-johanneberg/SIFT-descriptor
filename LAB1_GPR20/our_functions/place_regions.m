function region_centres = place_regions(centre, radius)

    region_centres = zeros(2,9); % creates empty position holder
    K = 2; % init scale factor, k < 2 make the regions overlap
    
    region_centres(:,1) = [centre(1) - radius*K ; centre(2) - radius*K];
    region_centres(:,2) = [centre(1) ; centre(2) - radius*K];
    region_centres(:,3) = [centre(1) + radius*K ; centre(2) - radius*K];
    region_centres(:,4) = [centre(1) - radius*K ; centre(2)];
    region_centres(:,5) = [centre(1) ; centre(2)];
    region_centres(:,6) = [centre(1) + radius*K ; centre(2)];
    region_centres(:,7) = [centre(1) - radius*K ; centre(2) + radius*K];
    region_centres(:,8) = [centre(1); centre(2) + radius*K];
    region_centres(:,9) = [centre(1) + radius*K ; centre(2) + radius*K];
    
end