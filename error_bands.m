% MANUALLY LOAD THE RESULTS FIRST
sample_method = 'linear';
main_path = fileparts(mfilename('fullpath'));
cd(main_path);
addpath(genpath(fullfile(main_path, 'gadgets')));

expectation_list = [ ...
"error_value_togtd_0", "error_value_togtd_400", "error_value_togtd_800", "error_value_togtd_900", ...
"error_value_togtd_950", "error_value_togtd_975", "error_value_togtd_990", "error_value_togtd_1000", ...   
"error_value_greedy", "error_value_mta_nonparam", "error_value_mta"];

% "error_value_totd_0", "error_value_totd_400", "error_value_totd_800", "error_value_totd_900", ...
% "error_value_totd_950", "error_value_totd_975", "error_value_totd_990", "error_value_totd_1000", ...

LineColors = [linspecer(numel(expectation_list) - 3); [1, 0, 0]; [0, 1, 0]; [0, 0, 1];];
num_points = 201;

CURVES = []; LEGENDS = {};
figure();
MIN = inf;
BANDWIDTH = 0.2;

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
        LEGEND = "META";
    elseif strcmp(result_name, "error_value_mta_nonparam")
        LEGEND = "META(np)";
    elseif strcmp(result_name, "error_value_togtd_0")
        LEGEND = "gtd(0)";
    elseif strcmp(result_name, "error_value_togtd_20")
        LEGEND = "gtd(.2)";
    elseif strcmp(result_name, "error_value_togtd_40") || strcmp(result_name, "error_value_togtd_400")
        LEGEND = "gtd(.4)";
    elseif strcmp(result_name, "error_value_togtd_60")
        LEGEND = "gtd(.6)";
    elseif strcmp(result_name, "error_value_togtd_80") || strcmp(result_name, "error_value_togtd_800")
        LEGEND = "gtd(.8)";
    elseif strcmp(result_name, "error_value_togtd_900")
        LEGEND = "gtd(.9)";
    elseif strcmp(result_name, "error_value_togtd_950")
        LEGEND = "gtd(.95)";
    elseif strcmp(result_name, "error_value_togtd_975")
        LEGEND = "gtd(.975)";
    elseif strcmp(result_name, "error_value_togtd_990")
        LEGEND = "gtd(.99)";
    elseif strcmp(result_name, "error_value_togtd_100") || strcmp(result_name, "error_value_togtd_1000")
        LEGEND = "gtd(1)";
    elseif strcmp(result_name, "error_value_totd_0")
        LEGEND = "td(0)";
    elseif strcmp(result_name, "error_value_totd_20")
        LEGEND = "td(.2)";
    elseif strcmp(result_name, "error_value_totd_40") || strcmp(result_name, "error_value_totd_400")
        LEGEND = "td(.4)";
    elseif strcmp(result_name, "error_value_totd_60")
        LEGEND = "td(.6)";
    elseif strcmp(result_name, "error_value_totd_80") || strcmp(result_name, "error_value_totd_800")
        LEGEND = "td(.8)";
    elseif strcmp(result_name, "error_value_totd_900")
        LEGEND = "td(.9)";
    elseif strcmp(result_name, "error_value_totd_950")
        LEGEND = "td(.95)";
    elseif strcmp(result_name, "error_value_totd_975")
        LEGEND = "td(.975)";
    elseif strcmp(result_name, "error_value_totd_990")
        LEGEND = "td(.99)";
    elseif strcmp(result_name, "error_value_totd_100") || strcmp(result_name, "error_value_totd_1000")
        LEGEND = "td(1)";
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
axis([1, 1000, MIN, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
xlabel('steps');
ylabel('MSE');

xticks([0 100 1000]);
xticklabels({'0', '10^5', '10^6'});

drawnow;