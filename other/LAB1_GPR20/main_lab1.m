%% Lab 1 SSY097

% Group 20
% Members: Victor Huke, Gudmundur Hjalmar Egilsson, Jonas Lecerof

%% Init 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%run this to initialize the folders, 
%make sure to be in folder "LAB1_GPR20" while running this
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc

addpath('stuff_for_lab_1')
addpath('stuff_for_lab_1\sift')
addpath('stuff_for_lab_1\church_test')
addpath('our_functions')

run prepare_digits.m % inits data for Ex 1.11 - 1.13

test_img = reshape((11:100), 10, 9); % creates a test image

%% Ex 1.1 (and Ex 1.3)
%Run this section to open the solution for ex 1.1 and ex 1.3
%The function is verified in Ex 1.2
open get_patch.m

%% Ex 1.2
%Run this section to verify that get_patch works correctly
patch_test = get_patch(test_img, 3, 3, 1);
imagesc(patch_test);
colormap gray

%% Ex 1.4
%Run this section to open the solution for ex 1.4
open gaussian_filter.m

%% Ex 1.5 
%Run this section to open the solution for ex 1.5
open gaussian_gradients.m

%% Ex 1.6
% Run this section and verify that the gradiants are correct
[grad_x,grad_y] = gaussian_gradients(test_img,3);
imagesc(test_img);
colormap gray
axis image
hold on
quiver(grad_x, grad_y);

%% Ex 1.7
% Run this section and verify that the histograms is correct
% open gradient_histogram.m
plot_bouquet(test_img,3);

%% Ex 1.8
% Run this section and verify that place_regions is working
% open place_regions.m
img = read_image('paper_with_digits.png');
centres = place_regions([605;600],30);
plot_squares(img, centres, 30);

%% Ex 1.9
% Run this section and verify that gradiant_descriptor is working
% open gradient_descriptor.m
img = read_image('paper_with_digits.png');
desc = gradient_descriptor(img, [600;600],25);

%% Ex 1.10
%Run this section to see the solution to Ex 1.10
%note: prepare_digits is executed in Init section
open prepare_digits.m

%% Ex 1.11
%Run this section to verify classify_digits.m 
%open classify_digit.m 
img_nr= 45; %Specifies image to classify (1-50)

label = classify_digit(digits_validation(img_nr).image,digits_training);

% Prints the answer
disp(['New try: image # ' int2str(img_nr)]);
disp(['Its classified as ' num2str(label)])
disp(['Correct Answer is ' num2str(digits_validation(img_nr).label)]);

%% Ex 1.12
%Run this section to verify solution to Ex 1.12 
correct_matches=0;% # of correct matches
for i = 1:1:length(digits_validation) % runs for the length of imgs to classify
    label1 = classify_digit(digits_validation(i).image,digits_training);
    label2 = digits_validation(i).label;
    if  label1 == label2 % compares the lables
        correct_matches = correct_matches + 1; % counts matches
    end
end

success_rate = correct_matches/length(digits_validation)*100;
disp(['Success rate ' num2str(success_rate) '%'])

%% Ex*1.13

%% Ex 1.14
% This code is jut to figure out how  extractSIFT and matchFeatures works
[coords_1, descs_1] = extractSIFT(digits_training(1).image);
[coords_2, descs_2] = extractSIFT(digits_validation(1).image);
corrs = matchFeatures(descs_1', descs_2', 'MatchThreshold', 100, 'MaxRatio', 0.7)
matching_points = size(corrs,1) 

%% Ex 1.15 (and Ex 1.16)
%Run this section to verify solution to Ex 1.15
% open classify_church.m
clc
load church_data.mat
load manual_labels.mat
correct = 0;

%checks all church images
for i = 1:1:10
    img_name = strcat('church',int2str(i),'.jpg');
    img = read_image(img_name);
    img = rgb2gray(img);
    % imagesc(img);
    name = classify_church(img,feature_collection);
    
    if  strcmp(char(manual_labels(i)),char(name))
        correct = correct + 1;
    else
        disp(['Wrong classification!!!!!!!!!!!!!']);
        disp([img_name]);
        disp(['Classified as church in ' char(name)]);
        disp(['The rigth answer is ' char(manual_labels(i))]);
    end
end

disp([int2str(correct) ' correct classifications']);


