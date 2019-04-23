expectation_list = ["togtd_0_results", ...
                    "togtd_50000_results", ...
                    "togtd_75000_results", ...
                    "togtd_87500_results", ...
                    "togtd_93750_results", ...
                    "togtd_96875_results", ...
                    "togtd_100000_results"];
LineColors = linspecer(numel(expectation_list));
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
    elseif strcmp(result_name, "togtd_0_results")
        LEGEND = "GTD(0)";
    elseif strcmp(result_name, "togtd_50000_results")
        LEGEND = "GTD(0.5)";
    elseif strcmp(result_name, "togtd_75000_results")
        LEGEND = "GTD(0.75)";
    elseif strcmp(result_name, "togtd_87500_results")
        LEGEND = "GTD(0.875)";
    elseif strcmp(result_name, "togtd_93750_results")
        LEGEND = "GTD(0.9375)";
    elseif strcmp(result_name, "togtd_96875_results")
        LEGEND = "GTD(0.96875)";
    elseif strcmp(result_name, "togtd_100000_results")
        LEGEND = "GTD(1)";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 12);
set(L, 'Location', 'east');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
drawnow;