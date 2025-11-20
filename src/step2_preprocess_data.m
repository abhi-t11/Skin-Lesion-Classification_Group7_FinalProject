clear; clc; close all;

%% Define paths
basePath = 'C:\Users\abhis\Documents\MATLAB\Skin Disease Detection'; % Change as needed
dataPath = fullfile(basePath, 'data');
organizedPath = fullfile(dataPath, 'organized_by_class');
resultsPath = fullfile(basePath, 'results');
figuresPath = fullfile(basePath, 'figures');

fprintf('=== Step 2: Data Preprocessing ===\n\n');

%% Create Image Datastore
fprintf('Creating image datastore...\n');

% Image input size for ResNet50 (standard size)
inputSize = [224 224 3];

% Create datastore from organized folders
imds = imageDatastore(organizedPath, ...
    'IncludeSubfolders', true, ...
    'LabelSource', 'foldernames');

fprintf('✓ Total images loaded: %d\n', numel(imds.Files));
fprintf('✓ Number of classes: %d\n', numel(categories(imds.Labels)));

%% Display class distribution
fprintf('\n--- Class Distribution ---\n');
labelCounts = countEachLabel(imds);
disp(labelCounts);

% Create visualization
figure('Position', [100 100 1000 600]);
bar(categorical(labelCounts.Label), labelCounts.Count);
ylabel('Number of Images', 'FontSize', 12);
xlabel('Diagnosis Category', 'FontSize', 12);
title('Dataset Distribution Before Splitting', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xtickangle(45);

% Add count labels
hold on;
for i = 1:height(labelCounts)
    text(i, labelCounts.Count(i) + 50, num2str(labelCounts.Count(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
hold off;

saveas(gcf, fullfile(figuresPath, 'step2_full_distribution.png'));
fprintf('✓ Saved distribution plot\n');

%% Split data into Train/Validation/Test
fprintf('\n--- Splitting Dataset ---\n');
fprintf('Split ratio: 70%% Train / 20%% Validation / 10%% Test\n');

% First split: 70% train, 30% remaining
[imdsTrain, imdsRemaining] = splitEachLabel(imds, 0.70, 'randomized');

% Second split: Split remaining into 20% val and 10% test
% 20/(20+10) = 0.67 of remaining = 20% of total
[imdsValidation, imdsTest] = splitEachLabel(imdsRemaining, 0.67, 'randomized');

% Display split statistics
fprintf('\nDataset split complete:\n');
fprintf('  Training set:   %d images (%.1f%%)\n', numel(imdsTrain.Files), ...
    (numel(imdsTrain.Files)/numel(imds.Files))*100);
fprintf('  Validation set: %d images (%.1f%%)\n', numel(imdsValidation.Files), ...
    (numel(imdsValidation.Files)/numel(imds.Files))*100);
fprintf('  Test set:       %d images (%.1f%%)\n', numel(imdsTest.Files), ...
    (numel(imdsTest.Files)/numel(imds.Files))*100);

% Show per-class distribution in each set
fprintf('\nPer-class distribution:\n');
trainCounts = countEachLabel(imdsTrain);
valCounts = countEachLabel(imdsValidation);
testCounts = countEachLabel(imdsTest);

splitTable = table(trainCounts.Label, trainCounts.Count, valCounts.Count, testCounts.Count, ...
    'VariableNames', {'Class', 'Train', 'Validation', 'Test'});
disp(splitTable);

% Save split table
writetable(splitTable, fullfile(resultsPath, 'data_split_summary.csv'));

%% Visualize sample images from training set
fprintf('\n--- Creating sample visualization ---\n');

figure('Position', [100 100 1400 900]);
sgtitle('Sample Training Images from Each Class', 'FontSize', 16, 'FontWeight', 'bold');

classes = categories(imdsTrain.Labels);
for i = 1:length(classes)
    % Find images of this class
    idx = find(imdsTrain.Labels == classes{i});
    
    % Show first 2 images from each class
    for j = 1:min(2, length(idx))
        subplot(length(classes), 2, (i-1)*2 + j);
        img = readimage(imdsTrain, idx(j));
        imshow(img);
        if j == 1
            ylabel(classes{i}, 'FontWeight', 'bold', 'Interpreter', 'none');
        end
        title(sprintf('Image %d', j), 'FontSize', 9);
    end
end

saveas(gcf, fullfile(figuresPath, 'step2_sample_training_images.png'));
fprintf('✓ Saved sample images visualization\n');

%% Define data augmentation for training
fprintf('\n--- Setting up data augmentation ---\n');

% Augmentation parameters (for training only)
imageAugmenter = imageDataAugmenter(...
    'RandRotation', [-20, 20], ...           % Random rotation ±20 degrees
    'RandXReflection', true, ...             % Random horizontal flip
    'RandYReflection', true, ...             % Random vertical flip
    'RandXScale', [0.8, 1.2], ...           % Random scaling 80-120%
    'RandYScale', [0.8, 1.2], ...           % Random scaling 80-120%
    'RandXShear', [-10, 10], ...            % Random shear
    'RandYShear', [-10, 10]);                % Random shear

fprintf('Augmentation settings:\n');
fprintf('  - Random rotation: ±20 degrees\n');
fprintf('  - Random flips: horizontal & vertical\n');
fprintf('  - Random scaling: 80-120%%\n');
fprintf('  - Random shear: ±10 degrees\n');

%% Create augmented image datastores
fprintf('\n--- Creating augmented datastores ---\n');

% Training set with augmentation
augmentedTrainDS = augmentedImageDatastore(inputSize, imdsTrain, ...
    'DataAugmentation', imageAugmenter, ...
    'ColorPreprocessing', 'gray2rgb', ...    % Convert grayscale to RGB if needed
    'OutputSizeMode', 'centercrop');

fprintf('✓ Training datastore created (with augmentation)\n');

% Validation set (no augmentation, only resizing)
augmentedValDS = augmentedImageDatastore(inputSize, imdsValidation, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'OutputSizeMode', 'centercrop');

fprintf('✓ Validation datastore created (no augmentation)\n');

% Test set (no augmentation, only resizing)
augmentedTestDS = augmentedImageDatastore(inputSize, imdsTest, ...
    'ColorPreprocessing', 'gray2rgb', ...
    'OutputSizeMode', 'centercrop');

fprintf('✓ Test datastore created (no augmentation)\n');

%%  Visualize augmentation effects
fprintf('\n--- Visualizing augmentation effects ---\n');

% Select sample images from first 3 classes
figure('Position', [100 100 1400 900]);
sgtitle('Data Augmentation Examples', 'FontSize', 16, 'FontWeight', 'bold');

classes = categories(imdsTrain.Labels);
plotCounter = 1;

for classIdx = 1:min(3, length(classes))  % Show first 3 classes
    % Find an image from this class
    idx = find(imdsTrain.Labels == classes{classIdx});
    sampleIdx = idx(1);
    
    % Read original image
    origImg = readimage(imdsTrain, sampleIdx);
    
    % Create temporary single-image datastore
    tempImds = imageDatastore(imdsTrain.Files{sampleIdx});
    
    % Create augmented datastore for this image
    tempAugDS = augmentedImageDatastore(inputSize, tempImds, ...
        'DataAugmentation', imageAugmenter, ...
        'OutputSizeMode', 'centercrop');
    
    % Show original (resized to target size)
    subplot(3, 4, plotCounter);
    origImgResized = imresize(origImg, [inputSize(1) inputSize(2)]);
    imshow(origImgResized);
    title(sprintf('%s - Original', classes{classIdx}), ...
        'Interpreter', 'none', 'FontWeight', 'bold', 'FontSize', 10);
    plotCounter = plotCounter + 1;
    
    % Show 3 augmented versions
    for augIdx = 1:3
        reset(tempAugDS);
        augData = read(tempAugDS);
        augImg = augData.input{1};
        
        subplot(3, 4, plotCounter);
        imshow(augImg);
        title(sprintf('Augmented %d', augIdx), 'FontSize', 10);
        plotCounter = plotCounter + 1;
    end
end

saveas(gcf, fullfile(figuresPath, 'step2_augmentation_examples.png'));
fprintf('✓ Saved augmentation visualization\n');

%% Calculate class weights (for handling imbalanced data)
fprintf('\n--- Calculating class weights ---\n');

% Get training set class distribution
trainLabelCounts = countEachLabel(imdsTrain);
numClasses = height(trainLabelCounts);
totalTrainSamples = sum(trainLabelCounts.Count);

% Calculate inverse frequency weights
classWeights = totalTrainSamples ./ (numClasses * trainLabelCounts.Count);

% Normalize weights
classWeights = classWeights / sum(classWeights) * numClasses;

fprintf('Class weights (for loss function):\n');
weightTable = table(trainLabelCounts.Label, trainLabelCounts.Count, classWeights, ...
    'VariableNames', {'Class', 'TrainCount', 'Weight'});
disp(weightTable);

% Visualize weights
figure('Position', [100 100 1000 600]);
bar(categorical(trainLabelCounts.Label), classWeights);
ylabel('Weight', 'FontSize', 12);
xlabel('Class', 'FontSize', 12);
title('Class Weights for Imbalanced Dataset', 'FontSize', 14, 'FontWeight', 'bold');
grid on;
xtickangle(45);

% Add value labels on bars
hold on;
for i = 1:length(classWeights)
    text(i, classWeights(i) + 0.05, sprintf('%.2f', classWeights(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 10, 'FontWeight', 'bold');
end
hold off;

saveas(gcf, fullfile(figuresPath, 'step2_class_weights.png'));
fprintf('✓ Saved class weights visualization\n');

%% Save all preprocessing results
fprintf('\n--- Saving preprocessing results ---\n');

preprocessData.inputSize = inputSize;
preprocessData.imdsTrain = imdsTrain;
preprocessData.imdsValidation = imdsValidation;
preprocessData.imdsTest = imdsTest;
preprocessData.augmentedTrainDS = augmentedTrainDS;
preprocessData.augmentedValDS = augmentedValDS;
preprocessData.augmentedTestDS = augmentedTestDS;
preprocessData.classWeights = classWeights;
preprocessData.trainLabelCounts = trainLabelCounts;
preprocessData.splitTable = splitTable;
preprocessData.classes = classes;
preprocessData.numClasses = numClasses;

% Save to file
save(fullfile(resultsPath, 'step2_preprocessed_data.mat'), 'preprocessData', '-v7.3');
fprintf('✓ Saved preprocessed data to: step2_preprocessed_data.mat\n');

%% Summary
fprintf('\n');
fprintf('====================================\n');
fprintf('   PREPROCESSING COMPLETE!         \n');
fprintf('====================================\n');
fprintf('Summary:\n');
fprintf('  - Input size: %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('  - Training samples: %d\n', numel(imdsTrain.Files));
fprintf('  - Validation samples: %d\n', numel(imdsValidation.Files));
fprintf('  - Test samples: %d\n', numel(imdsTest.Files));
fprintf('  - Number of classes: %d\n', numClasses);
fprintf('  - Data augmentation: Enabled for training\n');
fprintf('  - Class weights: Calculated\n');
fprintf('\nGenerated files:\n');
fprintf('  Results folder:\n');
fprintf('    - step2_preprocessed_data.mat\n');
fprintf('    - data_split_summary.csv\n');
fprintf('  Figures folder:\n');
fprintf('    - step2_full_distribution.png\n');
fprintf('    - step2_sample_training_images.png\n');
fprintf('    - step2_augmentation_examples.png\n');
fprintf('    - step2_class_weights.png\n');