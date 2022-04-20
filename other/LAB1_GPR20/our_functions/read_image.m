function img = read_image(file)
raw_image = imread(file);
img = im2double(raw_image);
end