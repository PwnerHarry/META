function u4alpha(env, folder, pattern)
% draw the u-shaped band, with x-axis as the learning rate alpha
% specify the configuration with pattern, using regular expression

% usage:
% ushaped('ringworld', 'ringworld/e_1e6_r_40', 'behavior\_0\.33\_target\_0\.25')
% ushaped('frozenlake', 'frozenlake/e_1e6_r_40', 'behavior\_0\.25\_target\_0\.2')

% get file list in which the files satisfy the filter
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
smoothing_window = 10;
if strcmp(env, 'ringworld')
    METHOD_LIST = {'totd_0', 'totd_400', 'totd_800', 'totd_900', 'totd_950', 'totd_975', 'totd_990', 'totd_1000', 'greedy', 'mta'};
elseif strcmp(env, 'frozenlake')
    METHOD_LIST = {'togtd_0', 'togtd_400', 'togtd_800', 'togtd_900', 'togtd_950', 'togtd_975', 'togtd_990', 'togtd_1000', 'greedy', 'mta_nonparam', 'mta'};
elseif strcmp(env, 'frozenlake_AC') || strcmp(env, 'mountaincar')
    METHOD_LIST = {'baseline_0', 'baseline_400', 'baseline_800', 'baseline_900', 'baseline_950', 'baseline_975', 'baseline_990', 'baseline_1000', 'greedy', 'MTA_nonparam', 'MTA'};
end
MEANS = nan(numel(METHOD_LIST), numel(filenames));
STDS = nan(numel(METHOD_LIST), numel(filenames));
ALPHAS = zeros(numel(filenames), 1);
for index_filename = 1: numel(filenames)
    filename = filenames{index_filename};
    if strcmp(env, 'ringworld')
        [startIndex, endIndex] = regexp(filename, 'a\_.*\_k');
        if isempty(startIndex) && isempty(endIndex)
            [startIndex, endIndex] = regexp(filename, 'a\_.*\_e');
        end
    elseif strcmp(env, 'frozenlake')
        [startIndex, endIndex] = regexp(filename, 'a\_.*\_b');
    elseif strcmp(env, 'frozenlake_AC') || strcmp(env, 'mountaincar')
        [startIndex, endIndex] = regexp(filename, 'a\_.*\_b');
    end
    alpha = str2double(filename(startIndex + 2: endIndex - 2));
    ALPHAS(index_filename) = alpha;
    loaded = load(fullfile(folder, filename)); %#ok<NASGU>
    for index_method = 1: numel(METHOD_LIST)
        method = METHOD_LIST{index_method};
        try
            if strcmp(env, 'mountaincar')
                eval(sprintf('MEANS(%d, index_filename) = -mean(loaded.return_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = -mean(loaded.return_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            else
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.error_value_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.error_value_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            end
        catch ME
        end
    end
end

[ALPHAS, IA, IC] = unique(ALPHAS);
NEW_MEANS = zeros(numel(METHOD_LIST), numel(IA));
NEW_STDS = zeros(numel(METHOD_LIST), numel(IA));
for index_unique = 1: numel(IA)
    locations = find(IC == index_unique);
    MEAN_MTA_nonparam = MEANS(end - 1, locations);
    MEAN_MTA = MEANS(end, locations);
    [~, IBEST_MTA] = min(MEAN_MTA);
    index_best = locations(IBEST_MTA);
    [~, IBEST_MTA_nonparam] = min(MEAN_MTA_nonparam);
    index_best_nonparam = locations(IBEST_MTA_nonparam);
    NEW_MEANS(end, index_unique) = MEANS(end, index_best);
    NEW_STDS(end, index_unique) = STDS(end, index_best);
    NEW_MEANS(end - 1, index_unique) = MEANS(end - 1, index_best_nonparam);
    NEW_STDS(end - 1, index_unique) = STDS(end - 1, index_best_nonparam);
    NEW_MEANS(1: end - 2, index_unique) = mean(MEANS(1: end - 2, locations), 2, 'omitnan');
    NEW_STDS(1: end - 2, index_unique) = mean(STDS(1: end - 2, locations), 2, 'omitnan');
end
MEANS = NEW_MEANS;
STDS = NEW_STDS;

[ALPHAS, I] = sort(ALPHAS, 'ascend');
MEANS = MEANS(:, I);
STDS = STDS(:, I);

% draw
main_path = fileparts(mfilename('fullpath'));
cd(main_path);
addpath(genpath(fullfile(main_path, 'gadgets')));
figure;
BANDWIDTH = 0.1;
LINECOLORS = [linspecer(numel(METHOD_LIST) - 3); [1, 0, 0]; [0, 1, 0]; [0, 0, 1];];
CURVES = []; LEGENDS = {};
for index_method = 1: numel(METHOD_LIST)
    MEAN = MEANS(index_method, :); STD = STDS(index_method, :);
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-STD; STD];
    [CURVE, ~] = band_drawer(ALPHAS', MEAN, INTERVAL, LINECOLORS(index_method, :));
    CURVES = [CURVES, CURVE];
    method = METHOD_LIST{index_method};
    if strcmp(method, "togtd_0") || strcmp(method, "baseline_0")
        LEGEND = "GTD(0)";
    elseif strcmp(method, "togtd_20") || strcmp(method, "baseline_20")
        LEGEND = "GTD(.2)";
    elseif strcmp(method, "togtd_40") || strcmp(method, "baseline_40") || strcmp(method, "togtd_400") || strcmp(method, "baseline_400")
        LEGEND = "GTD(.4)";
    elseif strcmp(method, "togtd_60") || strcmp(method, "baseline_60")
        LEGEND = "GTD(.6)";
    elseif strcmp(method, "togtd_80") || strcmp(method, "togtd_800") || strcmp(method, "baseline_80")
        LEGEND = "GTD(.8)";
    elseif strcmp(method, "togtd_90") || strcmp(method, "togtd_900") || strcmp(method, "baseline_90") || strcmp(method, "baseline_900")
        LEGEND = "GTD(.9)";
    elseif strcmp(method, "togtd_950") || strcmp(method, "baseline_950")
        LEGEND = "GTD(.95)";
    elseif strcmp(method, "togtd_975") || strcmp(method, "baseline_975")
        LEGEND = "GTD(.975)";
    elseif strcmp(method, "togtd_990") || strcmp(method, "baseline_990")
        LEGEND = "GTD(.99)";
    elseif strcmp(method, "togtd_100") || strcmp(method, "togtd_1000") || strcmp(method, "baseline_100")
        LEGEND = "GTD(1)";
    elseif strcmp(method, "totd_0")
        LEGEND = "TD(0)";
    elseif strcmp(method, "totd_20")
        LEGEND = "TD(.2)";
    elseif strcmp(method, "totd_40") || strcmp(method, "totd_400")
        LEGEND = "TD(.4)";
    elseif strcmp(method, "totd_60")
        LEGEND = "TD(.6)";
    elseif strcmp(method, "totd_80") || strcmp(method, "totd_800")
        LEGEND = "TD(.8)";
    elseif strcmp(method, "totd_90") || strcmp(method, "totd_900")
        LEGEND = "TD(.9)";
    elseif strcmp(method, "totd_950")
        LEGEND = "TD(.95)";
    elseif strcmp(method, "totd_975")
        LEGEND = "TD(.975)";
    elseif strcmp(method, "totd_990")
        LEGEND = "TD(.99)";
    elseif strcmp(method, "totd_100") || strcmp(method, "totd_1000")
        LEGEND = "TD(1)";
    elseif strcmp(method, "greedy")
        LEGEND = "greedy";
    elseif strcmp(method, "mta_nonparam") || strcmp(method, "MTA_nonparam")
        LEGEND = "MTA(np)";
    elseif strcmp(method, "mta") || strcmp(method, "MTA")
        LEGEND = "MTA";
    end
    LEGENDS = [LEGENDS, LEGEND];
end
L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([0, inf, -inf, inf]);
drawnow;
end