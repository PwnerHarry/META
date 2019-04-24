clear all;
folder = 'frozenlake';
policy = 'off';
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(strfind(filenames{i}, 'error_value')) || isempty(strfind(filenames{i}, policy))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];
for i = 1: length(filenames)
    filename = filenames{i};
    shelled_sample = load(fullfile(folder, filename));
    samples(i) = shelled_sample.sample;
end
CURVES = [];
LEGENDS = {};
for i = 1: length(samples)
    [CURVE, ~] = band_drawer(samples(i).X, samples(i).MEAN, samples(i).INTERVAL); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    LEGEND = samples(i).name;
    LEGEND = strrep(LEGEND, 'error_value_', '');
    LEGEND = strrep(LEGEND, policy, '');
    LEGEND = strrep(LEGEND, '_', ' ');
    LEGEND = strrep(LEGEND, 'mta ', 'MTA');
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 14);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([1, inf, 0, inf]);
drawnow;