expectation_list = ["error_value_togtd_0", ...
                    "error_value_togtd_20000", ...
                    "error_value_togtd_40000", ...
                    "error_value_togtd_60000", ...
                    "error_value_togtd_80000", ...
                    "error_value_togtd_100000", ...
                    "error_value_greedy", ...
                    "error_value_mta"];
LineColors = [linspecer(numel(expectation_list) - 2); [1, 0, 0]; [0, 0, 1];];
num_points = 201;

% filename = 'mta_N_11_behavior_0.15_target_0.05_episodes_5000.mat';
% load(filename);

CURVES = []; LEGENDS = {};
figure();

for result_index = 1: numel(expectation_list)
	result_name = expectation_list(result_index);
    try
        eval(sprintf('results = %s;', result_name))
    catch ME
        continue;
    end
    [X, MEAN, INTERVAL] = get_statistics(results, num_points);
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "error_value_mta")
        LEGEND = "MTA";
    elseif strcmp(result_name, "error_value_togtd_0")
        LEGEND = "GTD(0)";
    elseif strcmp(result_name, "error_value_togtd_20000")
        LEGEND = "GTD(0.2)";
    elseif strcmp(result_name, "error_value_togtd_40000")
        LEGEND = "GTD(0.4)";
    elseif strcmp(result_name, "error_value_togtd_60000")
        LEGEND = "GTD(0.6)";
    elseif strcmp(result_name, "error_value_togtd_80000")
        LEGEND = "GTD(0.8)";
    elseif strcmp(result_name, "error_value_togtd_100000")
        LEGEND = "GTD(1)";
    elseif strcmp(result_name, "error_value_greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 12);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([1, inf, 0, inf]);
drawnow;