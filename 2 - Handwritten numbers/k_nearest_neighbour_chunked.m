%% Nearest neighbour chunked
clear;
rawdata = load('Dataset/data.mat');

data.template_labels = rawdata.trainlab;
data.template_data = rawdata.trainv;
data.test_labels = rawdata.testlab;
data.test_data = rawdata.testv;

chunk_template_size = 6000;
chunk_test_size = 1000;

number_of_classes = 10;

% Modify k to swap between k-nearest neighbours and nearest neighbour
% (which is just with k = 1).
k = 1;

tic;
[confusion_matrix, classification_table] =...
    ChunkedKNearestNeighbours(data,...
                              k,...
                              number_of_classes,...
                              chunk_template_size,...
                              chunk_test_size);
toc;

%% Evaluate results
PrintResults(confusion_matrix);
PlotRandomImages(classification_table, data.test_data);