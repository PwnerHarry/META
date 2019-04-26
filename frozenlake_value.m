clear all;
figure;
folder = 'frozenlake';
policy = 'on';
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
    if ~isempty(strfind(samples(i).name, 'mta'))
        if ~isempty(strfind(samples(i).name, '0.5'))
            [CURVE, ~] = band_drawer(samples(i).X, samples(i).MEAN, samples(i).INTERVAL, [0, 0, 1]); %
        else
            continue;
        end
    elseif ~isempty(strfind(samples(i).name, 'greedy'))
        [CURVE, ~] = band_drawer(samples(i).X, samples(i).MEAN, samples(i).INTERVAL, [1, 0, 0]);
    else
        [CURVE, ~] = band_drawer(samples(i).X, samples(i).MEAN, samples(i).INTERVAL);
    end
    CURVES = [CURVES, CURVE];
    LEGEND = samples(i).name;
    LEGEND = strrep(LEGEND, 'error_value_', '');
    LEGEND = strrep(LEGEND, policy, '');
    LEGEND = strrep(LEGEND, '_', ' ');
    LEGEND = regexprep(LEGEND, 'mta.+', 'MTA');
    LEGEND = strrep(LEGEND, '  ', ' ');
    LEGEND = strrep(LEGEND, 'togtd 0', 'GTD(0)');
    LEGEND = strrep(LEGEND, 'togtd 20000', 'GTD(0.2)');
    LEGEND = strrep(LEGEND, 'togtd 40000', 'GTD(0.4)');
    LEGEND = strrep(LEGEND, 'togtd 60000', 'GTD(0.6)');
    LEGEND = strrep(LEGEND, 'togtd 80000', 'GTD(0.8)');
    LEGEND = strrep(LEGEND, 'togtd 100000', 'GTD(1)');
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([1, inf, 0, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;