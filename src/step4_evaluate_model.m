
clear; clc; close all;

%% Define paths
basePath = 'C:\Users\abhis\Documents\MATLAB\Skin Disease Detection'; % Change as needed
resultsPath = fullfile(basePath, 'results');
figuresPath = fullfile(basePath, 'figures');

fprintf('=== Step 4: Model Evaluation ===\n\n');

%% Load trained model and test data
fprintf('Loading trained model and test data...\n');

% Load model
load(fullfile(resultsPath, 'step3_trained_model.mat'), 'modelInfo');
trainedNet = modelInfo.trainedNet;
classes = modelInfo.classes;
numClasses = modelInfo.numClasses;

% Load preprocessed data
load(fullfile(resultsPath, 'step2_preprocessed_data.mat'), 'preprocessData');
augmentedTestDS = preprocessData.augmentedTestDS;
imdsTest = preprocessData.imdsTest;
trueLabels = imdsTest.Labels;

fprintf('✓ Model and data loaded\n');
fprintf('  Test samples: %d\n', numel(trueLabels));
fprintf('  Number of classes: %d\n', numClasses);

%% Make predictions on test set
fprintf('\n--- Making predictions on test set ---\n');
fprintf('This may take a few minutes...\n');

tic;
[predictedLabels, scores] = classify(trainedNet, augmentedTestDS);
predictionTime = toc;

fprintf('✓ Predictions complete\n');
fprintf('  Time taken: %.2f seconds\n', predictionTime);
fprintf('  Average time per image: %.3f seconds\n', predictionTime/numel(trueLabels));

%% Calculate overall accuracy
fprintf('\n--- Overall Performance ---\n');

overallAccuracy = sum(predictedLabels == trueLabels) / numel(trueLabels);
fprintf('Test Set Accuracy: %.2f%%\n', overallAccuracy * 100);

%% Calculate confusion matrix
fprintf('\n--- Generating confusion matrix ---\n');

confMat = confusionmat(trueLabels, predictedLabels);

% Visualize confusion matrix
figure('Position', [100 100 900 800]);
cm = confusionchart(trueLabels, predictedLabels);
cm.Title = 'Confusion Matrix - Test Set';
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
cm.FontSize = 11;
cm.DiagonalColor = [0.2 0.7 0.3];
cm.OffDiagonalColor = [0.8 0.2 0.2];

saveas(gcf, fullfile(figuresPath, 'step4_confusion_matrix.png'));
saveas(gcf, fullfile(figuresPath, 'step4_confusion_matrix.fig'));
fprintf('✓ Saved confusion matrix\n');

%% Calculate per-class metrics
fprintf('\n--- Per-Class Performance Metrics ---\n');

metricsTable = table();

fprintf('%-15s | Precision | Recall | F1-Score | Specificity | Support\n', 'Class');
fprintf('%s\n', repmat('-', 1, 75));

for i = 1:numClasses
    % Extract metrics
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    TN = sum(confMat(:)) - TP - FP - FN;
    
    % Calculate metrics
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
    
    if (TN + FP) > 0
        specificity = TN / (TN + FP);
    else
        specificity = 0;
    end
    
    support = sum(confMat(i,:));
    
    % Store in table
    metricsTable.Class{i} = char(classes(i));
    metricsTable.Precision(i) = precision;
    metricsTable.Recall(i) = recall;
    metricsTable.F1_Score(i) = f1;
    metricsTable.Specificity(i) = specificity;
    metricsTable.Support(i) = support;
    
    % Print
    fprintf('%-15s | %.4f    | %.4f | %.4f   | %.4f      | %d\n', ...
        classes{i}, precision, recall, f1, specificity, support);
end

% Calculate macro averages
fprintf('%s\n', repmat('-', 1, 75));
fprintf('%-15s | %.4f    | %.4f | %.4f   | %.4f      | %d\n', ...
    'Macro Average', ...
    mean(metricsTable.Precision), ...
    mean(metricsTable.Recall), ...
    mean(metricsTable.F1_Score), ...
    mean(metricsTable.Specificity), ...
    sum(metricsTable.Support));

% Calculate weighted averages
weightedPrecision = sum(metricsTable.Precision .* metricsTable.Support) / sum(metricsTable.Support);
weightedRecall = sum(metricsTable.Recall .* metricsTable.Support) / sum(metricsTable.Support);
weightedF1 = sum(metricsTable.F1_Score .* metricsTable.Support) / sum(metricsTable.Support);

fprintf('%-15s | %.4f    | %.4f | %.4f   | -          | %d\n', ...
    'Weighted Avg', ...
    weightedPrecision, ...
    weightedRecall, ...
    weightedF1, ...
    sum(metricsTable.Support));

% Save metrics table
writetable(metricsTable, fullfile(resultsPath, 'step4_detailed_metrics.csv'));
fprintf('\n✓ Saved detailed metrics to CSV\n');

%% Visualize per-class performance
fprintf('\n--- Creating performance comparison chart ---\n');

figure('Position', [100 100 1200 600]);

% Create grouped bar chart
x = 1:numClasses;
y = [metricsTable.Precision, metricsTable.Recall, metricsTable.F1_Score];

b = bar(x, y);
b(1).FaceColor = [0.2 0.4 0.8];
b(2).FaceColor = [0.8 0.4 0.2];
b(3).FaceColor = [0.3 0.7 0.3];

% Formatting
xlabel('Class', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Score', 'FontSize', 12, 'FontWeight', 'bold');
title('Per-Class Performance Metrics', 'FontSize', 14, 'FontWeight', 'bold');
legend({'Precision', 'Recall', 'F1-Score'}, 'Location', 'southoutside', 'Orientation', 'horizontal');
set(gca, 'XTick', x, 'XTickLabel', classes, 'XTickLabelRotation', 45);
grid on;
ylim([0 1]);

saveas(gcf, fullfile(figuresPath, 'step4_performance_comparison.png'));
fprintf('✓ Saved performance comparison chart\n');

%% ROC Curves and AUC
fprintf('\n--- Generating ROC curves ---\n');

figure('Position', [100 100 1400 900]);

aucScores = zeros(numClasses, 1);

for i = 1:numClasses
    % Binary labels for this class (one-vs-all)
    binaryLabels = double(trueLabels == classes{i});
    classScores = scores(:, i);
    
    % Calculate ROC curve
    [X, Y, ~, AUC] = perfcurve(binaryLabels, classScores, 1);
    aucScores(i) = AUC;
    
    % Plot
    subplot(3, 3, i);
    plot(X, Y, 'LineWidth', 2.5, 'Color', [0.2 0.4 0.8]);
    hold on;
    plot([0 1], [0 1], '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5); % Diagonal line
    hold off;
    
    xlabel('False Positive Rate', 'FontSize', 10);
    ylabel('True Positive Rate', 'FontSize', 10);
    title(sprintf('%s\nAUC = %.3f', classes{i}, AUC), ...
        'Interpreter', 'none', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    axis square;
    xlim([0 1]);
    ylim([0 1]);
end

sgtitle('ROC Curves (One-vs-All)', 'FontSize', 16, 'FontWeight', 'bold');

saveas(gcf, fullfile(figuresPath, 'step4_roc_curves.png'));
saveas(gcf, fullfile(figuresPath, 'step4_roc_curves.fig'));
fprintf('✓ Saved ROC curves\n');

% Add AUC to metrics table
metricsTable.AUC = aucScores;

% Print AUC scores
fprintf('\nAUC Scores:\n');
for i = 1:numClasses
    fprintf('  %s: %.4f\n', classes{i}, aucScores(i));
end
fprintf('  Mean AUC: %.4f\n', mean(aucScores));

%% Visualize correct predictions
fprintf('\n--- Creating correct predictions visualization ---\n');

% Find correctly classified samples
correctIdx = find(predictedLabels == trueLabels);

% Randomly select samples from each class
figure('Position', [100 100 1400 900]);
sgtitle('Correctly Classified Examples', 'FontSize', 16, 'FontWeight', 'bold');

plotCounter = 1;
for i = 1:numClasses
    % Find correct predictions for this class
    classCorrectIdx = correctIdx(trueLabels(correctIdx) == classes{i});
    
    if ~isempty(classCorrectIdx)
        % Select up to 2 random samples
        numSamples = min(2, length(classCorrectIdx));
        selectedIdx = classCorrectIdx(randperm(length(classCorrectIdx), numSamples));
        
        for j = 1:numSamples
            if plotCounter <= 14  % 7 classes × 2 samples
                subplot(7, 2, plotCounter);
                
                % Read and display image
                img = imread(imdsTest.Files{selectedIdx(j)});
                imshow(img);
                
                % Get confidence
                confidence = max(scores(selectedIdx(j), :)) * 100;
                
                % Title
                titleStr = sprintf('%s\nConfidence: %.1f%%', ...
                    char(classes(i)), confidence);
                title(titleStr, 'Interpreter', 'none', 'FontSize', 9, ...
                    'FontWeight', 'bold', 'Color', [0 0.7 0]);
                
                plotCounter = plotCounter + 1;
            end
        end
    end
end

saveas(gcf, fullfile(figuresPath, 'step4_correct_predictions.png'));
fprintf('✓ Saved correct predictions visualization\n');

%% Visualize incorrect predictions (Error Analysis)
fprintf('\n--- Creating error analysis visualization ---\n');

% Find incorrectly classified samples
incorrectIdx = find(predictedLabels ~= trueLabels);

fprintf('Total misclassifications: %d (%.2f%%)\n', ...
    length(incorrectIdx), (length(incorrectIdx)/numel(trueLabels))*100);

% Select random misclassified samples
numErrorSamples = min(16, length(incorrectIdx));
errorSampleIdx = incorrectIdx(randperm(length(incorrectIdx), numErrorSamples));

figure('Position', [100 100 1400 1000]);
sgtitle('Misclassified Examples - Error Analysis', 'FontSize', 16, 'FontWeight', 'bold');

for i = 1:numErrorSamples
    subplot(4, 4, i);
    
    % Read and display image
    img = imread(imdsTest.Files{errorSampleIdx(i)});
    imshow(img);
    
    % Get true and predicted labels
    trueLabel = trueLabels(errorSampleIdx(i));
    predLabel = predictedLabels(errorSampleIdx(i));
    confidence = max(scores(errorSampleIdx(i), :)) * 100;
    
    % Add red border
    hold on;
    rectangle('Position', [1 1 size(img,2)-1 size(img,1)-1], ...
        'EdgeColor', 'r', 'LineWidth', 3);
    hold off;
    
    % Title with error info
    titleStr = sprintf('True: %s\nPred: %s (%.1f%%)', ...
        char(trueLabel), char(predLabel), confidence);
    title(titleStr, 'Interpreter', 'none', 'FontSize', 8, ...
        'FontWeight', 'bold', 'Color', [0.8 0 0]);
end

saveas(gcf, fullfile(figuresPath, 'step4_incorrect_predictions.png'));
fprintf('✓ Saved error analysis visualization\n');

%% Confusion pair analysis
fprintf('\n--- Most Common Confusion Pairs ---\n');

% Find most common misclassifications
confMatNodiag = confMat;
confMatNodiag(logical(eye(size(confMat)))) = 0; % Remove diagonal

[sortedErrors, idx] = sort(confMatNodiag(:), 'descend');
topN = min(5, sum(sortedErrors > 0));

fprintf('Top %d confusion pairs:\n', topN);
for i = 1:topN
    [row, col] = ind2sub(size(confMat), idx(i));
    fprintf('  %d. %s → %s: %d cases (%.1f%% of %s samples)\n', ...
        i, char(classes(row)), char(classes(col)), sortedErrors(i), ...
        (sortedErrors(i)/sum(confMat(row,:)))*100, char(classes(row)));
end

%% Save all results
fprintf('\n--- Saving evaluation results ---\n');

evaluationResults.overallAccuracy = overallAccuracy;
evaluationResults.confusionMatrix = confMat;
evaluationResults.predictedLabels = predictedLabels;
evaluationResults.trueLabels = trueLabels;
evaluationResults.scores = scores;
evaluationResults.metricsTable = metricsTable;
evaluationResults.aucScores = aucScores;
evaluationResults.predictionTime = predictionTime;
evaluationResults.incorrectIdx = incorrectIdx;
evaluationResults.correctIdx = correctIdx;

save(fullfile(resultsPath, 'step4_evaluation_results.mat'), 'evaluationResults', '-v7.3');
fprintf('✓ Saved evaluation results to MAT file\n');

% Create summary table for report
summaryTable = table();
summaryTable.Metric = {
    'Overall Accuracy';
    'Macro-Average Precision';
    'Macro-Average Recall';
    'Macro-Average F1-Score';
    'Weighted-Average Precision';
    'Weighted-Average Recall';
    'Weighted-Average F1-Score';
    'Mean AUC';
    'Total Samples';
    'Correct Predictions';
    'Incorrect Predictions';
    'Prediction Time (s)'
};

summaryTable.Value = [
    overallAccuracy;
    mean(metricsTable.Precision);
    mean(metricsTable.Recall);
    mean(metricsTable.F1_Score);
    weightedPrecision;
    weightedRecall;
    weightedF1;
    mean(aucScores);
    numel(trueLabels);
    length(correctIdx);
    length(incorrectIdx);
    predictionTime
];

writetable(summaryTable, fullfile(resultsPath, 'step4_summary_metrics.csv'));
fprintf('✓ Saved summary metrics to CSV\n');

%%  Final summary
fprintf('\n');
fprintf('====================================\n');
fprintf('   EVALUATION COMPLETE!            \n');
fprintf('====================================\n');
fprintf('Test Set Performance:\n');
fprintf('  Overall Accuracy: %.2f%%\n', overallAccuracy * 100);
fprintf('  Macro-Average F1: %.4f\n', mean(metricsTable.F1_Score));
fprintf('  Weighted-Average F1: %.4f\n', weightedF1);
fprintf('  Mean AUC: %.4f\n', mean(aucScores));
fprintf('  Correct predictions: %d / %d\n', length(correctIdx), numel(trueLabels));
fprintf('  Incorrect predictions: %d / %d\n', length(incorrectIdx), numel(trueLabels));
fprintf('\nBest performing classes (by F1-Score):\n');
[~, sortIdx] = sort(metricsTable.F1_Score, 'descend');
for i = 1:3
    fprintf('  %d. %s: %.4f\n', i, metricsTable.Class{sortIdx(i)}, ...
        metricsTable.F1_Score(sortIdx(i)));
end
fprintf('\nWorst performing classes (by F1-Score):\n');
for i = numClasses:-1:max(1, numClasses-2)
    fprintf('  %d. %s: %.4f\n', numClasses-i+1, metricsTable.Class{sortIdx(i)}, ...
        metricsTable.F1_Score(sortIdx(i)));
end
fprintf('\nGenerated files:\n');
fprintf('  Results folder:\n');
fprintf('    - step4_evaluation_results.mat\n');
fprintf('    - step4_detailed_metrics.csv\n');
fprintf('    - step4_summary_metrics.csv\n');
fprintf('  Figures folder:\n');
fprintf('    - step4_confusion_matrix.png\n');
fprintf('    - step4_performance_comparison.png\n');
fprintf('    - step4_roc_curves.png\n');
fprintf('    - step4_correct_predictions.png\n');
fprintf('    - step4_incorrect_predictions.png\n');