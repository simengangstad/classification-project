%% Nearest neighbour without chunks or clustering
% This script will take quite a long time to finish, taking a coffee break
% is adviced :)

clear;
rawdata = load('Dataset/data.mat');

data.template_labels = rawdata.trainlab;
data.template_data = rawdata.trainv;
data.test_labels = rawdata.testlab;
data.test_data = rawdata.testv;

number_of_classes = 10;

% Modify k to swap between k-nearest neighbours and nearest neighbour
% (which is just with k = 1).
k = 7;

tic;
disp('Running nearest neighbour...');
[confusion_matrix, classification_table] =...
    KNearestNeighbours(data, k, number_of_classes);
toc;

%% Evaluate results
PrintResults(confusion_matrix);
PlotRandomImages(classification_table, data.test_data);