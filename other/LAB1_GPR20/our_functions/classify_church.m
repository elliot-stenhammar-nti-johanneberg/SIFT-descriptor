function name = classify_church(img,features)

[coords, descs_1] = extractSIFT(img); % Creates discriptor for features in the image

%Match 
matches = matchFeatures(descs_1', features.descriptors', 'MatchThreshold', 100, 'MaxRatio', 0.7);
labels = zeros(size(matches,1),1);
% for i = 1:1:size(matches,1)
    labels(:) = features.labels(matches(:,2));
% end
name = features.names(mode(labels));
end