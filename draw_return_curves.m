% MANUALLY LOAD THE RESULTS FIRST
sample_method = 'log';
cd(fileparts(mfilename('fullpath'))); addpath(genpath(cd));

expectation_list = [ ...
    "return"];
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
    MEAN = results_mean(X);
    MIN = min(min(MEAN), MIN);
    INTERVAL = repmat(MEAN, 2, 1) + BANDWIDTH * [-results_std(X); results_std(X)];
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