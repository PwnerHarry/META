% MANUALLY LOAD THE RESULTS FIRST
sample_method = 'log';
cd(fileparts(mfilename('fullpath'))); addpath(genpath(cd));

expectation_list = [ ...
    "error_value_totd_0", "error_value_totd_20", "error_value_totd_40", ...
    "error_value_totd_60", "error_value_totd_80", "error_value_totd_100", ...
    "error_value_togtd_0", "error_value_togtd_20", "error_value_togtd_40", ...
    "error_value_togtd_60", "error_value_togtd_80", "error_value_togtd_100", ...
    "error_value_greedy", "error_value_mta"];
LineColors = [linspecer(numel(expectation_list) - 2); [1, 0, 0]; [0, 0, 1];];
num_points = 201;

CURVES = []; LEGENDS = {};
figure();
MIN = inf;
BANDWIDTH = 0.05;

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
    MEAN(1) = mean(results_mean(X(1): round(0.5 * X(1) + 0.5 * X(2))));
    STD(1) = mean(results_std(X(1): round(0.5 * X(1) + 0.5 * X(2))));
    for i = 2: length(X) - 1
        MEAN(i) = mean(results_mean(round(0.5 * X(i - 1) + 0.5 * X(i)): round(0.5 * X(i) + 0.5 * X(i + 1))));
        STD(i) = mean(results_std(round(0.5 * X(i - 1) + 0.5 * X(i)): round(0.5 * X(i) + 0.5 * X(i + 1))));
    end
    MEAN(end) = mean(results_mean(round(0.5 * X(end - 1) + 0.5 * X(end)): X(end)));
    STD(end) = mean(results_std(round(0.5 * X(end - 1) + 0.5 * X(end)): X(end)));
    MIN = min(min(MEAN), MIN);
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-STD; STD];
    INTERVAL(INTERVAL <= 0) = eps;
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "error_value_mta")
        LEGEND = "MTA";
    elseif strcmp(result_name, "error_value_togtd_0")
        LEGEND = "GTD(0)";
    elseif strcmp(result_name, "error_value_togtd_20")
        LEGEND = "GTD(.2)";
    elseif strcmp(result_name, "error_value_togtd_40")
        LEGEND = "GTD(.4)";
    elseif strcmp(result_name, "error_value_togtd_60")
        LEGEND = "GTD(.6)";
    elseif strcmp(result_name, "error_value_togtd_80")
        LEGEND = "GTD(.8)";
    elseif strcmp(result_name, "error_value_togtd_100")
        LEGEND = "GTD(1)";
    elseif strcmp(result_name, "error_value_totd_0")
        LEGEND = "TD(0)";
    elseif strcmp(result_name, "error_value_totd_20")
        LEGEND = "TD(.2)";
    elseif strcmp(result_name, "error_value_totd_40")
        LEGEND = "TD(.4)";
    elseif strcmp(result_name, "error_value_totd_60")
        LEGEND = "TD(.6)";
    elseif strcmp(result_name, "error_value_totd_80")
        LEGEND = "TD(.8)";
    elseif strcmp(result_name, "error_value_totd_100")
        LEGEND = "TD(1)";
    elseif strcmp(result_name, "error_value_greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
if strcmp(sample_method, 'log')
    set(gca, 'xscale', 'log');
end
set(gca, 'yscale', 'log');
axis([1, inf, MIN, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;