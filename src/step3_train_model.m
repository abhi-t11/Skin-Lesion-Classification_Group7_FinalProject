clear; clc; close all;

%% Define paths
basePath = 'C:\Users\abhis\Documents\MATLAB\Skin Disease Detection'; % Change as needed
resultsPath = fullfile(basePath, 'results');
figuresPath = fullfile(basePath, 'figures');

fprintf('=== Step 3: Model Training ===\n\n');

%%  Load preprocessed data
fprintf('Loading preprocessed data...\n');
load(fullfile(resultsPath, 'step2_preprocessed_data.mat'), 'preprocessData');

% Extract variables
augmentedTrainDS = preprocessData.augmentedTrainDS;
augmentedValDS = preprocessData.augmentedValDS;
augmentedTestDS = preprocessData.augmentedTestDS;
classWeights = preprocessData.classWeights;
classes = preprocessData.classes;
numClasses = preprocessData.numClasses;
inputSize = preprocessData.inputSize;

fprintf('✓ Data loaded successfully\n');
fprintf('  Training samples: %d\n', numel(preprocessData.imdsTrain.Files));
fprintf('  Validation samples: %d\n', numel(preprocessData.imdsValidation.Files));
fprintf('  Test samples: %d\n', numel(preprocessData.imdsTest.Files));
fprintf('  Number of classes: %d\n', numClasses);

%% Load pretrained ResNet50
fprintf('\n--- Loading pretrained ResNet50 ---\n');

% Load ResNet50 (pretrained on ImageNet)
net = resnet50;
fprintf('✓ ResNet50 loaded\n');

% Display network architecture info
fprintf('  Input size: %s\n', mat2str(net.Layers(1).InputSize));
fprintf('  Total layers: %d\n', numel(net.Layers));
fprintf('  Original classes: 1000 (ImageNet)\n');
fprintf('  Target classes: %d (HAM10000)\n', numClasses);

%% Modify network for transfer learning
fprintf('\n--- Modifying network architecture ---\n');

% Convert to layer graph
lgraph = layerGraph(net);

% Display layer names to find the correct ones
fprintf('Analyzing ResNet50 architecture...\n');

% Get all layer names
layerNames = {lgraph.Layers.Name};

% Find the classification layer (last layer)
lastLayerName = layerNames{end};
fprintf('  Last layer name: %s\n', lastLayerName);

% Find the fully connected layer
% ResNet50 typically has 'fc1000' but let's search for it
fcLayerIdx = [];
for i = 1:numel(lgraph.Layers)
    if isa(lgraph.Layers(i), 'nnet.cnn.layer.FullyConnectedLayer')
        fcLayerIdx = [fcLayerIdx, i];
    end
end

% Get the last FC layer (the one before classification)
if ~isempty(fcLayerIdx)
    fcLayerName = layerNames{fcLayerIdx(end)};
    fprintf('  FC layer to replace: %s\n', fcLayerName);
else
    error('Could not find fully connected layer in ResNet50');
end

% Replace fully connected layer
newFC = fullyConnectedLayer(numClasses, ...
    'Name', 'new_fc', ...
    'WeightLearnRateFactor', 10, ...     % Learn faster for new layer
    'BiasLearnRateFactor', 10);

lgraph = replaceLayer(lgraph, fcLayerName, newFC);

% Replace classification layer
newClassLayer = classificationLayer('Name', 'new_classoutput');
lgraph = replaceLayer(lgraph, lastLayerName, newClassLayer);

fprintf('✓ Network modified for %d classes\n', numClasses);
fprintf('  - Replaced %s → new_fc (%d outputs)\n', fcLayerName, numClasses);
fprintf('  - Replaced %s → new_classoutput\n', lastLayerName);

%% Set training options
fprintf('\n--- Setting training options ---\n');

% Check if GPU is available
if canUseGPU()
    executionEnv = 'gpu';
    fprintf('✓ GPU detected - training will use GPU acceleration\n');
    try
        gpuInfo = gpuDevice;
        fprintf('  GPU: %s\n', gpuInfo.Name);
    catch
        fprintf('  GPU: Available\n');
    end
else
    executionEnv = 'cpu';
    fprintf('⚠ No GPU detected - training will use CPU (slower)\n');
end

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 16, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedValDS, ...
    'ValidationFrequency', 50, ...
    'ValidationPatience', 5, ...
    'Verbose', true, ...
    'VerboseFrequency', 50, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', executionEnv, ...
    'OutputNetwork', 'best-validation-loss');

fprintf('Training configuration:\n');
fprintf('  Optimizer: Adam\n');
fprintf('  Learning rate: %.0e\n', options.InitialLearnRate);
fprintf('  Max epochs: %d\n', options.MaxEpochs);
fprintf('  Mini-batch size: %d\n', options.MiniBatchSize);
fprintf('  Validation frequency: every %d iterations\n', options.ValidationFrequency);
fprintf('  Early stopping: Yes (patience = %d)\n', options.ValidationPatience);
fprintf('  Execution: %s\n', upper(executionEnv));

%% Train the network
fprintf('\n--- Starting training ---\n');
fprintf('This may take 15-60 minutes depending on your hardware...\n');
fprintf('A training progress window will appear.\n\n');

% Start timer
trainingStartTime = tic;

% Train the network
try
    trainedNet = trainNetwork(augmentedTrainDS, lgraph, options);
    trainingTime = toc(trainingStartTime);
    
    fprintf('\n✓ Training completed successfully!\n');
    fprintf('  Total training time: %.2f minutes\n', trainingTime/60);
    
catch ME
    fprintf('\n✗ Training failed with error:\n');
    fprintf('  %s\n', ME.message);
    error('Training terminated. Check error above.');
end

%% Save trained model
fprintf('\n--- Saving trained model ---\n');

modelInfo.trainedNet = trainedNet;
modelInfo.trainingTime = trainingTime;
modelInfo.trainingOptions = options;
modelInfo.classes = classes;
modelInfo.numClasses = numClasses;
modelInfo.inputSize = inputSize;
modelInfo.classWeights = classWeights;

save(fullfile(resultsPath, 'step3_trained_model.mat'), 'modelInfo', '-v7.3');
fprintf('✓ Model saved to: step3_trained_model.mat\n');

%% Evaluate on validation set
fprintf('\n--- Quick validation set evaluation ---\n');

% Predict on validation set
[valPredictions, valScores] = classify(trainedNet, augmentedValDS);
valLabels = preprocessData.imdsValidation.Labels;

% Calculate validation accuracy
valAccuracy = sum(valPredictions == valLabels) / numel(valLabels);
fprintf('Validation Accuracy: %.2f%%\n', valAccuracy * 100);

% Confusion matrix for validation set
figure('Position', [100 100 800 700]);
cm = confusionchart(valLabels, valPredictions);
cm.Title = 'Confusion Matrix - Validation Set';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.FontSize = 10;

saveas(gcf, fullfile(figuresPath, 'step3_validation_confusion_matrix.png'));
fprintf('✓ Saved validation confusion matrix\n');

%% Per-class performance on validation set
fprintf('\n--- Per-class validation performance ---\n');

confMat = confusionmat(valLabels, valPredictions);

fprintf('%-15s | Precision | Recall | F1-Score\n', 'Class');
fprintf('%s\n', repmat('-', 1, 55));

for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
    
    fprintf('%-15s | %.4f    | %.4f | %.4f\n', ...
        classes{i}, precision, recall, f1);
end

%% Visualize sample predictions
fprintf('\n--- Creating sample prediction visualizations ---\n');

% Get random samples from validation set
numSamples = 12;
valFiles = preprocessData.imdsValidation.Files;
sampleIndices = randperm(numel(valFiles), numSamples);

figure('Position', [100 100 1400 900]);
sgtitle('Sample Predictions on Validation Set', 'FontSize', 16, 'FontWeight', 'bold');

for i = 1:numSamples
    subplot(3, 4, i);
    
    % Read image
    img = imread(valFiles{sampleIndices(i)});
    imshow(img);
    
    % Get true and predicted labels
    trueLabel = valLabels(sampleIndices(i));
    predLabel = valPredictions(sampleIndices(i));
    confidence = max(valScores(sampleIndices(i), :)) * 100;
    
    % Color code: green if correct, red if wrong
    if trueLabel == predLabel
        titleColor = [0 0.7 0];
        borderColor = 'g';
    else
        titleColor = [0.9 0 0];
        borderColor = 'r';
    end
    
    % Add colored border
    hold on;
    rectangle('Position', [1 1 size(img,2)-1 size(img,1)-1], ...
        'EdgeColor', borderColor, 'LineWidth', 3);
    hold off;
    
    % Title with prediction info
    titleStr = sprintf('True: %s\nPred: %s (%.1f%%)', ...
        char(trueLabel), char(predLabel), confidence);
    title(titleStr, 'Color', titleColor, 'FontSize', 9, ...
        'Interpreter', 'none', 'FontWeight', 'bold');
end

saveas(gcf, fullfile(figuresPath, 'step3_sample_predictions.png'));
fprintf('✓ Saved sample predictions visualization\n');

%% Training summary
fprintf('\n');
fprintf('====================================\n');
fprintf('     TRAINING COMPLETE!            \n');
fprintf('====================================\n');
fprintf('Model Summary:\n');
fprintf('  Architecture: ResNet50 (Transfer Learning)\n');
fprintf('  Training time: %.2f minutes\n', trainingTime/60);
fprintf('  Validation accuracy: %.2f%%\n', valAccuracy * 100);
fprintf('  Number of classes: %d\n', numClasses);
fprintf('  Input size: %dx%dx%d\n', inputSize(1), inputSize(2), inputSize(3));
fprintf('\nSaved files:\n');
fprintf('  - step3_trained_model.mat (model + metadata)\n');
fprintf('  - step3_validation_confusion_matrix.png\n');
fprintf('  - step3_sample_predictions.png\n');