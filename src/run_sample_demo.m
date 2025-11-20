%% Quick Demo Using Sample Dataset
% This script demonstrates the project using the included sample dataset
clear; clc; close all;

fprintf('========================================\n');
fprintf('  QUICK DEMO - SAMPLE DATASET          \n');
fprintf('========================================\n\n');

fprintf('This demo uses the sample dataset (70 images)\n');
fprintf('for quick verification that the code works.\n\n');

fprintf('Expected runtime: 2-3 minutes\n');
fprintf('Expected accuracy: 50-60% (normal for small dataset)\n\n');

%% Check if sample dataset exists
basePath = fileparts(pwd);  % Go up one directory from src/
samplePath = fullfile(basePath, 'data', 'sample_dataset');

if ~exist(samplePath, 'dir')
    error(['Sample dataset not found!\n' ...
           'Expected location: %s\n' ...
           'Please ensure sample_dataset folder exists.'], samplePath);
end

fprintf('✓ Sample dataset found\n\n');

%% Temporarily modify paths to use sample dataset
fprintf('Setting up sample data paths...\n');

% This is a simplified version that only does classification
% without requiring the full preprocessing pipeline

%% Load sample images
fprintf('Loading sample images...\n');

imds = imageDatastore(samplePath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('✓ Loaded %d images\n', numel(imds.Files));
fprintf('✓ Found %d classes\n\n', numel(categories(imds.Labels)));

%% Simple train/test split (80/20)
fprintf('Splitting data (80%% train, 20%% test)...\n');
[imdsTrain, imdsTest] = splitEachLabel(imds, 0.8, 'randomized');

fprintf('  Training: %d images\n', numel(imdsTrain.Files));
fprintf('  Test: %d images\n\n', numel(imdsTest.Files));

%% Create augmented datastores
fprintf('Preparing data...\n');
inputSize = [224 224 3];

augmenter = imageDataAugmenter(...
    'RandRotation', [-20, 20], ...
    'RandXReflection', true, ...
    'RandYReflection', true);

augTrainDS = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', augmenter);
augTestDS = augmentedImageDatastore(inputSize, imdsTest);

fprintf('✓ Data prepared\n\n');

%% Load and modify ResNet50
fprintf('Loading ResNet50...\n');
net = resnet50;
numClasses = numel(categories(imdsTrain.Labels));

lgraph = layerGraph(net);
layers = lgraph.Layers;
layerNames = {layers.Name};

% Find FC and classification layers
fcLayerIdx = find(contains(layerNames, 'fc', 'IgnoreCase', true));
fcLayerName = layerNames{fcLayerIdx(end)};
classLayerName = layerNames{end};

% Replace layers
newFC = fullyConnectedLayer(numClasses, 'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, 'BiasLearnRateFactor', 10);
lgraph = replaceLayer(lgraph, fcLayerName, newFC);

newClassLayer = classificationLayer('Name', 'new_class');
lgraph = replaceLayer(lgraph, classLayerName, newClassLayer);

fprintf('✓ Model prepared for %d classes\n\n', numClasses);

%% Training options (reduced for speed)
fprintf('Setting up training (reduced epochs for demo)...\n');

options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 5, ...  % Reduced from 30
    'MiniBatchSize', 8, ...  % Smaller batch for small dataset
    'ValidationData', augTestDS, ...
    'ValidationFrequency', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

fprintf('✓ Training configured\n\n');

%% Train
fprintf('Starting training (this will take 2-3 minutes)...\n');
fprintf('A training progress window will appear.\n\n');

tic;
trainedNet = trainNetwork(augTrainDS, lgraph, options);
trainingTime = toc;

fprintf('✓ Training complete in %.2f minutes\n\n', trainingTime/60);

%% Evaluate
fprintf('Evaluating on test set...\n');
[predictions, scores] = classify(trainedNet, augTestDS);
trueLabels = imdsTest.Labels;

accuracy = sum(predictions == trueLabels) / numel(trueLabels);

fprintf('✓ Test Accuracy: %.2f%%\n\n', accuracy * 100);

%% Show sample predictions
fprintf('Displaying sample predictions...\n');

figure('Position', [100 100 1000 600]);
numSamples = 6;
randIdx = randperm(numel(imdsTest.Files), numSamples);

for i = 1:numSamples
    subplot(2, 3, i);
    img = readimage(imdsTest, randIdx(i));
    imshow(img);
    
    pred = predictions(randIdx(i));
    true = trueLabels(randIdx(i));
    conf = max(scores(randIdx(i), :)) * 100;
    
    if pred == true
        color = [0 0.7 0];
    else
        color = [0.9 0 0];
    end
    
    title(sprintf('True: %s\nPred: %s (%.1f%%)', ...
        char(true), char(pred), conf), ...
        'Color', color, 'Interpreter', 'none');
end
sgtitle('Sample Predictions', 'FontSize', 14, 'FontWeight', 'bold');

%% Summary
fprintf('\n');
fprintf('========================================\n');
fprintf('  DEMO COMPLETE!                       \n');
fprintf('========================================\n');
fprintf('Results Summary:\n');
fprintf('  - Training time: %.2f minutes\n', trainingTime/60);
fprintf('  - Test accuracy: %.2f%%\n', accuracy * 100);
fprintf('  - Dataset: Sample (70 images)\n\n');

fprintf('Note: These results are from a tiny sample dataset.\n');
fprintf('For full results (!83%% accuracy), download the complete\n');
fprintf('HAM10000 dataset following instructions in data_readme.txt\n\n');