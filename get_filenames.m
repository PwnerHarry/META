function filenames = get_filenames(folder, pattern)
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(regexp(filenames{i}, pattern, 'once'))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];
end
