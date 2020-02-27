function u4alpha(env, folder, pattern)
% draw the u-shaped band, with x-axis as the learning rate alpha
% specify the configuration with pattern, using regular expression

% usage:
% ushaped('ringworld', 'ringworld', '.*.mat')
% ushaped('frozenlake', 'frozenlake', '.*.mat')

main_path = fileparts(mfilename('fullpath'));
cd(main_path);
addpath(genpath(fullfile(main_path, 'gadgets')));

if strcmp(env, 'ringworld')
    METHOD_LIST = {'totd_0', 'totd_400', 'totd_800', 'totd_900', 'totd_950', 'totd_975', 'totd_990', 'totd_1000', 'greedy', 'mta'};
    smoothing_window = 100;
elseif strcmp(env, 'frozenlake')
    METHOD_LIST = {'togtd_0', 'togtd_400', 'togtd_800', 'togtd_900', 'togtd_950', 'togtd_975', 'togtd_990', 'togtd_1000', 'greedy', 'mta_nonparam', 'mta'};
    smoothing_window = 25;
elseif strcmp(env, 'mountaincar')
    METHOD_LIST = {'baseline_0', 'baseline_400', 'baseline_800', 'baseline_900', 'baseline_950', 'baseline_975', 'greedy', 'MTA'}; % 'baseline_1000', 
    smoothing_window = 100;
end

[ALPHAS_UNI, LAMBDAS_UNI, KAPPAS_UNI, table_baseline, table_greedy, table_META, table_META_np] = get_summary(env, folder, pattern);
%% draw baselines & greedy vs. best META
figure;
BANDWIDTH = 0.2;
if strcmp(env, 'mountaincar') || strcmp(env, 'ringworld')
    LINECOLORS = [linspecer(numel(METHOD_LIST) - 2); [1, 0, 0]; [0, 0, 1];];
else
    LINECOLORS = [linspecer(numel(METHOD_LIST) - 3); [1, 0, 0]; [0, 1, 0]; [0, 0, 1];];
end
CURVES = []; LEGENDS = {};
% draw baselines
for index_lambda = 1: length(LAMBDAS_UNI)
    lambda = LAMBDAS_UNI(index_lambda);
    if strcmp(env, 'mountaincar') && lambda == 1
        continue;
    end
    MEAN = table_baseline(:, index_lambda, 1);
    STD = table_baseline(:, index_lambda, 2);
    INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
    [CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, LINECOLORS(index_lambda, :), 1);
    CURVES = [CURVES, CURVE];
    LEGENDS = [LEGENDS, get_legend(METHOD_LIST{index_lambda})];
end
% draw greedy
MEAN = table_greedy(:, 1, 1);
STD = table_greedy(:, 1, 2);
INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
[CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, [1, 0, 0], 2);
CURVES = [CURVES, CURVE];
LEGENDS = [LEGENDS, 'greedy'];
% draw best META_np
if strcmp(env, 'frozenlake') 
    MEANS = table_META_np(:, :, 1);
    STDS = table_META_np(:, :, 2);
    [MEAN, args_best] = min(MEANS, [], 2);
    STD = NaN(size(MEAN));
    for i = 1: size(STDS, 1)
        STD(i) = STDS(i, args_best(i));
    end
    INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
    [CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, [0, 1, 0], 2);
    CURVES = [CURVES, CURVE];
    LEGENDS = [LEGENDS, 'METAnp'];
end
% draw best META
MEANS = table_META(:, :, 1);
STDS = table_META(:, :, 2);
if strcmp(env, 'mountaincar')
    [MEAN, args_best] = max(MEANS, [], 2);
else
    [MEAN, args_best] = min(MEANS, [], 2);
end
STD = NaN(size(MEAN));
for i = 1: size(STDS, 1)
    STD(i) = STDS(i, args_best(i));
end
INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
[CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, [0, 0, 1], 2);
CURVES = [CURVES, CURVE];
LEGENDS = [LEGENDS, 'META'];

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(gca, 'xscale', 'log');
axis([0, inf, -inf, inf]);
if ~strcmp(env, 'mountaincar')
    set(gca, 'yscale', 'log');
end
xlabel('learning rate');
if strcmp(env, 'mountaincar')
    ylabel('return');
else
    ylabel('MSE');
end
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;


%% draw METAs & greedy vs. best baseline
figure;
BANDWIDTH = 0.2;
if strcmp(env, 'mountaincar') || strcmp(env, 'ringworld')
    LINECOLORS = [linspecer(length(KAPPAS_UNI)); [1, 0, 0]; [0, 0, 1];];
else
    LINECOLORS = [linspecer(length(KAPPAS_UNI)); [1, 0, 0]; [0, 1, 0]; [0, 0, 1];];
end
CURVES = []; LEGENDS = {};
% draw METAs
for index_kappa = 1: length(KAPPAS_UNI)
    MEAN = table_META(:, index_kappa, 1);
    STD = table_META(:, index_kappa, 2);
    INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
    [CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, LINECOLORS(index_kappa, :), 1);
    CURVES = [CURVES, CURVE];
    LEGEND = sprintf('M(%.0e)', KAPPAS_UNI(index_kappa));
    LEGEND(end - 2) = []; % to get rid of the 0
    LEGENDS = [LEGENDS, LEGEND];
end
% draw greedy
MEAN = table_greedy(:, 1, 1);
STD = table_greedy(:, 1, 2);
INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
[CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, [1, 0, 0], 2);
CURVES = [CURVES, CURVE];
LEGENDS = [LEGENDS, 'greedy'];
% draw best baseline
if strcmp(env, 'mountaincar')
MEANS = table_baseline(:, 1: end - 1, 1);
STDS = table_baseline(:, 1: end - 1, 2);
else
MEANS = table_baseline(:, :, 1);
STDS = table_baseline(:, :, 2);
end
if strcmp(env, 'mountaincar')
    [MEAN, args_best] = max(MEANS, [], 2);
else
    [MEAN, args_best] = min(MEANS, [], 2);
end
STD = NaN(size(MEAN));
for i = 1: size(STDS, 1)
    STD(i) = STDS(i, args_best(i));
end
INTERVAL = repmat(MEAN, 1, 2) + BANDWIDTH * [-STD, STD];
[CURVE, ~] = band_drawer(ALPHAS_UNI, MEAN, INTERVAL, [0.1, 0.1, 0.1], 2);
CURVES = [CURVES, CURVE];
if strcmp(env, 'ringworld')
    LEGEND = "td*";
else
    LEGEND = "gtd*";
end
LEGENDS = [LEGENDS, LEGEND];
L = legend(CURVES, LEGENDS);

set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(gca, 'xscale', 'log');
axis([0, inf, -inf, inf]);
if ~strcmp(env, 'mountaincar')
    set(gca, 'yscale', 'log');
end
xlabel('learning rate');
if strcmp(env, 'mountaincar')
    ylabel('return');
else
    ylabel('MSE');
end
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;
%%%%%%%
tMETA_MEAN = table_META(:, :, 1);
tMETA_STD = table_META(:, :, 2);
[m, I] = min(tMETA_MEAN, [], 2);
if strcmp(env, 'frozenlake')
    tMETAnp_MEAN = table_META_np(:, :, 1);
    tMETAnp_STD = table_META_np(:, :, 2);
    [mnp, Inp] = min(tMETAnp_MEAN, [], 2);
end
%%%%%%%
end