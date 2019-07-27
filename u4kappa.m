function u4kappa(env, folder, pattern)
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(regexp(filenames{i}, pattern, 'once'))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];

% loading data files one by one
smoothing_window = 1000;
if strcmp(env, 'ringworld')
    METHOD_LIST = {'mta'};
elseif strcmp(env, 'frozenlake')
    METHOD_LIST = {'mta'};
else
    METHOD_LIST = {'baseline_0', 'baseline_20', 'baseline_40', 'baseline_60', 'baseline_80', 'baseline_100', 'greedy', 'MTA'};
end
MEANS = nan(numel(METHOD_LIST), numel(filenames));
STDS = nan(numel(METHOD_LIST), numel(filenames));
KAPPAS = zeros(numel(filenames), 1);
for index_filename = 1: numel(filenames)
    filename = filenames{index_filename};
    [startIndex, endIndex] = regexp(filename, 'k\_.*\_e');
    kappa = str2double(filename(startIndex + 2: endIndex - 2));
    KAPPAS(index_filename) = kappa;
    loaded = load(fullfile(folder, filename)); %#ok<NASGU>
    for index_method = 1: numel(METHOD_LIST)
        method = METHOD_LIST{index_method};
        try
            if strcmp(env, 'frozenlake_AC')
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.return_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.return_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            else
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.error_value_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.error_value_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            end
        catch ME
        end
    end
end

[KAPPAS, I] = sort(KAPPAS, 'ascend');
MEANS = MEANS(:, I);
STDS = STDS(:, I);

% draw
cd(fileparts(mfilename('fullpath'))); addpath(genpath(cd));
figure;
BANDWIDTH = 0.5;
LINECOLORS = [0, 0, 1];
CURVES = []; LEGENDS = {};
for index_method = 1: numel(METHOD_LIST)
    MEAN = MEANS(index_method, :); STD = STDS(index_method, :);
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-STD; STD];
    if strcmp(method, 'mta') || strcmp(method, 'MTA')
        LEGEND = "MTA";
        reduce_index = find(isnan(KAPPAS));
        KAPPAS(reduce_index) = [];
        MEAN(:, reduce_index) = [];
        INTERVAL(:, reduce_index) = [];
    end
    try
        [CURVE, ~] = band_drawer(KAPPAS', MEAN, INTERVAL, LINECOLORS(index_method, :));
    catch ME
        continue;
    end
    CURVES = [CURVES, CURVE];
    method = METHOD_LIST{index_method};
if strcmp(method, "togtd_0") || strcmp(method, "baseline_0")
        LEGEND = "GTD(0)";
    elseif strcmp(method, "togtd_20") || strcmp(method, "baseline_20")
        LEGEND = "GTD(.2)";
    elseif strcmp(method, "togtd_40") || strcmp(method, "baseline_40")
        LEGEND = "GTD(.4)";
    elseif strcmp(method, "togtd_60") || strcmp(method, "baseline_60")
        LEGEND = "GTD(.6)";
    elseif strcmp(method, "togtd_80") || strcmp(method, "baseline_80")
        LEGEND = "GTD(.8)";
    elseif strcmp(method, "togtd_100") || strcmp(method, "baseline_100")
        LEGEND = "GTD(1)";
    elseif strcmp(method, "totd_0")
        LEGEND = "TD(0)";
    elseif strcmp(method, "totd_20")
        LEGEND = "TD(.2)";
    elseif strcmp(method, "totd_40")
        LEGEND = "TD(.4)";
    elseif strcmp(method, "totd_60")
        LEGEND = "TD(.6)";
    elseif strcmp(method, "totd_80")
        LEGEND = "TD(.8)";
    elseif strcmp(method, "totd_100")
        LEGEND = "TD(1)";
    elseif strcmp(method, "greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end
L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
drawnow;
end