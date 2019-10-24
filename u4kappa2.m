function u4kappa2(env, folder, pattern)
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(regexp(filenames{i}, pattern, 'once'))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];

main_path = fileparts(mfilename('fullpath'));
cd(main_path);
addpath(genpath(fullfile(main_path, 'gadgets')));

% loading data files one by one
smoothing_window = 100;
if strcmp(env, 'ringworld')
    METHOD_LIST = {'mta'};
    COMPARED_LIST = {'totd_0', 'totd_400', 'totd_800', 'totd_900', 'totd_950', 'totd_975', 'totd_990', 'totd_1000', 'greedy'};
elseif strcmp(env, 'frozenlake')
    METHOD_LIST = {'mta'};
else
    METHOD_LIST = {'MTA'};
end

compared_performance = NaN(numel(COMPARED_LIST), 2); % 1st column the means and 2nd the stds!

MEANS = nan(numel(METHOD_LIST), numel(filenames));
STDS = nan(numel(METHOD_LIST), numel(filenames));
KAPPAS = NaN(numel(filenames), 1);
for index_filename = 1: numel(filenames)
    filename = filenames{index_filename};
    [startIndex, endIndex] = regexp(filename, 'k\_.*\_e');
    if isempty(startIndex) && isempty(endIndex)
        loaded = load(fullfile(folder, filename)); %#ok<NASGU>
        for index_compared_method = 1: numel(COMPARED_LIST)
            compared_method = COMPARED_LIST{index_compared_method};
            try
                eval(sprintf('compared_performance(index_compared_method, 1) = mean(loaded.error_value_%s_mean(end - %d: end), ''omitnan'');', compared_method, smoothing_window));
                eval(sprintf('compared_performance(index_compared_method, 2) = mean(loaded.error_value_%s_std(end - %d: end), ''omitnan'');', compared_method, smoothing_window));
            catch ME
                print('what happened?')
            end
        end
        continue;
    end
    kappa = str2double(filename(startIndex + 2: endIndex - 2));
    KAPPAS(index_filename) = kappa;
    loaded = load(fullfile(folder, filename)); %#ok<NASGU>
    for index_method = 1: numel(METHOD_LIST)
        method = METHOD_LIST{index_method};
        try
            if strcmp(env, 'mountaincar')
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.return_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.return_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            elseif strcmp(env, 'cartpole') || strcmp(env, 'frozenlake_AC')
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
figure;
BANDWIDTH = 0.05;
LINECOLORS = [[0, 0, 1]; linspecer(numel(COMPARED_LIST) - 1); [1, 0, 0]];
CURVES = []; LEGENDS = {};
for index_method = 1: numel(METHOD_LIST)
    method = METHOD_LIST{index_method};
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
    if strcmp(method, "togtd_0") || strcmp(method, "baseline_0")
        LEGEND = "gtd(0)";
    elseif strcmp(method, "togtd_20") || strcmp(method, "baseline_20")
        LEGEND = "gtd(.2)";
    elseif strcmp(method, "togtd_40") || strcmp(method, "baseline_40") || strcmp(method, "togtd_400") || strcmp(method, "baseline_400")
        LEGEND = "gtd(.4)";
    elseif strcmp(method, "togtd_60") || strcmp(method, "baseline_60")
        LEGEND = "gtd(.6)";
    elseif strcmp(method, "togtd_80") || strcmp(method, "togtd_800") || strcmp(method, "baseline_80") || strcmp(method, "baseline_800")
        LEGEND = "gtd(.8)";
    elseif strcmp(method, "togtd_90") || strcmp(method, "togtd_900") || strcmp(method, "baseline_90") || strcmp(method, "baseline_900")
        LEGEND = "gtd(.9)";
    elseif strcmp(method, "togtd_950") || strcmp(method, "baseline_950")
        LEGEND = "gtd(.95)";
    elseif strcmp(method, "togtd_975") || strcmp(method, "baseline_975")
        LEGEND = "gtd(.975)";
    elseif strcmp(method, "togtd_990") || strcmp(method, "baseline_990")
        LEGEND = "gtd(.99)";
    elseif strcmp(method, "togtd_100") || strcmp(method, "togtd_1000") || strcmp(method, "baseline_100") || strcmp(method, "baseline_1000")
        LEGEND = "gtd(1)";
    elseif strcmp(method, "totd_0")
        LEGEND = "td(0)";
    elseif strcmp(method, "totd_20")
        LEGEND = "td(.2)";
    elseif strcmp(method, "totd_40") || strcmp(method, "totd_400")
        LEGEND = "td(.4)";
    elseif strcmp(method, "totd_60")
        LEGEND = "td(.6)";
    elseif strcmp(method, "totd_80") || strcmp(method, "totd_800")
        LEGEND = "td(.8)";
    elseif strcmp(method, "totd_90") || strcmp(method, "totd_900")
        LEGEND = "td(.9)";
    elseif strcmp(method, "totd_950")
        LEGEND = "td(.95)";
    elseif strcmp(method, "totd_975")
        LEGEND = "td(.975)";
    elseif strcmp(method, "totd_990")
        LEGEND = "td(.99)";
    elseif strcmp(method, "totd_100") || strcmp(method, "totd_1000")
        LEGEND = "td(1)";
    elseif strcmp(method, "greedy")
        LEGEND = "greedy";
    elseif strcmp(method, "mta_nonparam") || strcmp(method, "MTA_nonparam")
        LEGEND = "META(np)";
    elseif strcmp(method, "mta") || strcmp(method, "MTA")
        LEGEND = "META";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

for index_compared_method = 1: numel(COMPARED_LIST)
    method = COMPARED_LIST{index_compared_method};
    MEAN = compared_performance(index_compared_method, 1);
    STD = compared_performance(index_compared_method, 2); 
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-STD; STD];
    INTERVAL = [INTERVAL, INTERVAL];
    MEAN = [MEAN, MEAN];
    STD = [STD, STD];
    [CURVE, ~] = band_drawer([min(KAPPAS(KAPPAS > 0)), max(KAPPAS)], MEAN, INTERVAL, LINECOLORS(index_compared_method + 1, :));
    CURVES = [CURVES, CURVE];
    if strcmp(method, "togtd_0") || strcmp(method, "baseline_0")
        LEGEND = "gtd(0)";
    elseif strcmp(method, "togtd_20") || strcmp(method, "baseline_20")
        LEGEND = "gtd(.2)";
    elseif strcmp(method, "togtd_40") || strcmp(method, "baseline_40") || strcmp(method, "togtd_400") || strcmp(method, "baseline_400")
        LEGEND = "gtd(.4)";
    elseif strcmp(method, "togtd_60") || strcmp(method, "baseline_60")
        LEGEND = "gtd(.6)";
    elseif strcmp(method, "togtd_80") || strcmp(method, "togtd_800") || strcmp(method, "baseline_80") || strcmp(method, "baseline_800")
        LEGEND = "gtd(.8)";
    elseif strcmp(method, "togtd_90") || strcmp(method, "togtd_900") || strcmp(method, "baseline_90") || strcmp(method, "baseline_900")
        LEGEND = "gtd(.9)";
    elseif strcmp(method, "togtd_950") || strcmp(method, "baseline_950")
        LEGEND = "gtd(.95)";
    elseif strcmp(method, "togtd_975") || strcmp(method, "baseline_975")
        LEGEND = "gtd(.975)";
    elseif strcmp(method, "togtd_990") || strcmp(method, "baseline_990")
        LEGEND = "gtd(.99)";
    elseif strcmp(method, "togtd_100") || strcmp(method, "togtd_1000") || strcmp(method, "baseline_100") || strcmp(method, "baseline_1000")
        LEGEND = "gtd(1)";
    elseif strcmp(method, "totd_0")
        LEGEND = "td(0)";
    elseif strcmp(method, "totd_20")
        LEGEND = "td(.2)";
    elseif strcmp(method, "totd_40") || strcmp(method, "totd_400")
        LEGEND = "td(.4)";
    elseif strcmp(method, "totd_60")
        LEGEND = "td(.6)";
    elseif strcmp(method, "totd_80") || strcmp(method, "totd_800")
        LEGEND = "td(.8)";
    elseif strcmp(method, "totd_90") || strcmp(method, "totd_900")
        LEGEND = "td(.9)";
    elseif strcmp(method, "totd_950")
        LEGEND = "td(.95)";
    elseif strcmp(method, "totd_975")
        LEGEND = "td(.975)";
    elseif strcmp(method, "totd_990")
        LEGEND = "td(.99)";
    elseif strcmp(method, "totd_100") || strcmp(method, "totd_1000")
        LEGEND = "td(1)";
    elseif strcmp(method, "greedy")
        LEGEND = "greedy";
    elseif strcmp(method, "mta_nonparam") || strcmp(method, "MTA_nonparam")
        LEGEND = "META(np)";
    elseif strcmp(method, "mta") || strcmp(method, "MTA")
        LEGEND = "META";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([min(KAPPAS(KAPPAS > 0)), max(KAPPAS), 0, inf])
xlabel('\kappa');
ylabel('MSE');
drawnow;



end