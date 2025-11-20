%% Simple script to organize images by class
clear; clc;

%% Ultra-Simple Merge Script
clear; clc;

dataPath = 'C:\Users\abhis\Documents\MATLAB\Skin Disease Detection\data'; % Change as needed
part1Path = fullfile(dataPath, 'HAM10000_images_part_1');
part2Path = fullfile(dataPath, 'HAM10000_images_part_2');
mergedPath = fullfile(dataPath, 'all_images');

% Create merged folder
if ~exist(mergedPath, 'dir')
    mkdir(mergedPath);
end

% Copy all from part 1
fprintf('Copying from Part 1...\n');
copyfile(fullfile(part1Path, '*.jpg'), mergedPath);

% Copy all from part 2
fprintf('Copying from Part 2...\n');
copyfile(fullfile(part2Path, '*.jpg'), mergedPath);

fprintf('Done! All images merged into: %s\n', mergedPath);

% Verify
mergedFiles = dir(fullfile(mergedPath, '*.jpg'));
fprintf('Total images: %d\n', length(mergedFiles));

dataPath = 'C:\Users\abhis\Documents\MATLAB\Skin Disease Detection\data'; % Change as needed
allImagesPath = fullfile(dataPath, 'all_images');
organizedPath = fullfile(dataPath, 'organized_by_class');
metadataFile = fullfile(dataPath, 'HAM10000_metadata.csv');

% Read metadata
metadata = readtable(metadataFile);

% Copy images to class folders
fprintf('Organizing %d images...\n', height(metadata));

for i = 1:height(metadata)
    imageId = metadata.image_id{i};
    className = metadata.dx{i};
    
    sourceFile = fullfile(allImagesPath, [imageId, '.jpg']);
    destFile = fullfile(organizedPath, className, [imageId, '.jpg']);
    
    if exist(sourceFile, 'file')
        copyfile(sourceFile, destFile);
    end
    
    if mod(i, 500) == 0
        fprintf('Progress: %d/%d (%.1f%%)\n', i, height(metadata), (i/height(metadata))*100);
    end
end

fprintf('Done! Images organized by class.\n');