% MANUALLY LOAD THE RESULTS FIRST
sample_method = 'linear';
main_path = fileparts(mfilename('fullpath'));
cd(main_path);
addpath(genpath(fullfile(main_path, 'gadgets')));

expectation_list = [ ...
    "return_baseline_0", "return_baseline_20", "return_baseline_40", ...
    "return_baseline_60", "return_baseline_80", "return_baseline_100", ...
    "return_baseline_400", "return_baseline_800", "return_baseline_900", ...
    "return_baseline_950", "return_baseline_975", ... % "return_baseline_990", ...
    "return_greedy", "return_MTA_nonparam", "return_MTA"]; %"return_baseline_1000", 
LineColors = [linspecer(numel(expectation_list) - 3); [1, 0, 0]; [0, 1, 0]; [0, 0, 1];];
num_points = 101;

CURVES = []; LEGENDS = {};
figure();
MIN = inf;
BANDWIDTH = 0.1;

for result_index = 1: numel(expectation_list)
    result_name  = expectation_list(result_index);
    try
        eval(sprintf('results_mean = %s_mean;', result_name));
        eval(sprintf('results_std = %s_std;', result_name));
    catch ME
        continue;
    end
    if strcmp(sample_method, 'linear')
        X = get_statistics2(results_mean, num_points, false);
    elseif strcmp(sample_method, 'log')
        X = get_statistics2(results_mean, num_points, true);
    end
    MEAN = NaN(size(X)); STD = NaN(size(X));
%     X_non_nan = intersect(find(~isnan(results_mean)), X);
%     MEAN = interp1(X_non_nan, results_mean(X_non_nan), X);
%     STD = interp1(X_non_nan, results_std(X_non_nan), X);
    MEAN(1) = mean(results_mean(X(1): round(0.5 * X(1) + 0.5 * X(2))), 'omitnan');
    STD(1) = mean(results_std(X(1): round(0.5 * X(1) + 0.5 * X(2))), 'omitnan');
    for i = 2: length(X) - 1
        MEAN(i) = mean(results_mean(round(0.5 * X(i - 1) + 0.5 * X(i)): round(0.5 * X(i) + 0.5 * X(i + 1))), 'omitnan');
        STD(i) = mean(results_std(round(0.5 * X(i - 1) + 0.5 * X(i)): round(0.5 * X(i) + 0.5 * X(i + 1))), 'omitnan');
    end
    MEAN(end) = mean(results_mean(round(0.5 * X(end - 1) + 0.5 * X(end)): X(end)), 'omitnan');
    STD(end) = mean(results_std(round(0.5 * X(end - 1) + 0.5 * X(end)): X(end)), 'omitnan');
    MIN = min(min(MEAN), MIN);
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-STD; STD];
    % INTERVAL(INTERVAL <= 0) = eps;
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "return_mta") || strcmp(result_name, "return_MTA")
        LEGEND = "META";
    elseif strcmp(result_name, "return_mta_nonparam") || strcmp(result_name, "return_MTA_nonparam")
        LEGEND = "MTA(np)";
    elseif strcmp(result_name, "return_baseline_0")
        LEGEND = "gtd(0)";
    elseif strcmp(result_name, "return_baseline_20")
        LEGEND = "gtd(.2)";
    elseif strcmp(result_name, "return_baseline_40") || strcmp(result_name, "return_baseline_400")
        LEGEND = "gtd(.4)";
    elseif strcmp(result_name, "return_baseline_60")
        LEGEND = "gtd(.6)";
    elseif strcmp(result_name, "return_baseline_80") || strcmp(result_name, "return_baseline_800")
        LEGEND = "gtd(.8)";
    elseif strcmp(result_name, "return_baseline_90") || strcmp(result_name, "return_baseline_900")
        LEGEND = "gtd(.9)";
    elseif strcmp(result_name, "return_baseline_95") || strcmp(result_name, "return_baseline_950")
        LEGEND = "gtd(.95)";
    elseif strcmp(result_name, "return_baseline_975")
        LEGEND = "gtd(.975)";
    elseif strcmp(result_name, "return_baseline_100") || strcmp(result_name, "return_baseline_1000")
        LEGEND = "gtd(1)";
    elseif strcmp(result_name, "return_greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
if strcmp(sample_method, 'log')
    set(gca, 'xscale', 'log');
end
% set(gca, 'yscale', 'log');
axis([1, inf, MIN, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
xlabel('steps');
ylabel('return');
xticks([0 100 200]);
xticklabels({'0', '25000', '50000'});
drawnow;