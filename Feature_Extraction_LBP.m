%Mallab version: R2017b

clc;
clear all;
close all;

imgSets = imageSet('/home/soumen/Desktop/TML_project/FERA-2013/', 'recursive');
feature = [];
for k = 1:length(imgSets)
    for l = 1:imgSets(k).Count
        Input_image = read(imgSets(k), l);
        if size(Input_image, 3) == 3
            Input_image = rgb2gray(Input_image);
        end
        [m, n] = size(Input_image);
        F = [];
        m1 = floor(m/8) * 8;
        n1 = floor(n/8) * 8;
        for i = 1 : floor(m/8) : m1
            for j = 1 : floor(n/8) : n1
                lbp_feature = extractLBPFeatures(Input_image(i:floor(i+m/8-1), j:floor(j+n/8-1)));
                F = horzcat(F, lbp_feature);
            end
        end
        F = horzcat(F, k);
        feature = [feature; F];
    end
end
csvwrite('/home/soumen/Desktop/TML_project/FERA-2013/originalFaceFeature_FERA-13.csv', feature);
