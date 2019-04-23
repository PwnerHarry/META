expectation_list = ["off_togtd_00_results", ...
                    "off_togtd_02_results", ...
                    "off_togtd_04_results", ...
                    "off_togtd_06_results", ...
                    "off_togtd_08_results", ...
                    "off_togtd_10_results", ...
                    "direct_greedy_results", ...
                    "error_value"];
LineColors = [linspecer(numel(expectation_list) - 2); [1, 0, 0]; [0, 0, 1];];
num_points = 201;

% filename = 'mta_N_11_behavior_0.15_target_0.05_episodes_5000.mat';
% load(filename);

CURVES = []; LEGENDS = {};
figure();

for result_index = 1: numel(expectation_list)
   result_name  = expectation_list(result_index);
    eval(sprintf('results = %s;', result_name))
    [X, MEAN, INTERVAL] = get_statistics(results, num_points);
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "error_value")
        LEGEND = "MTA";
    elseif strcmp(result_name, "off_togtd_00_results")
        LEGEND = "GTD(0)";
    elseif strcmp(result_name, "off_togtd_02_results")
        LEGEND = "GTD(0.2)";
    elseif strcmp(result_name, "off_togtd_04_results")
        LEGEND = "GTD(0.4)";
    elseif strcmp(result_name, "off_togtd_06_results")
        LEGEND = "GTD(0.6)";
    elseif strcmp(result_name, "off_togtd_08_results")
        LEGEND = "GTD(0.8)";
    elseif strcmp(result_name, "off_togtd_10_results")
        LEGEND = "GTD(1)";
    elseif strcmp(result_name, "direct_greedy_results")
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