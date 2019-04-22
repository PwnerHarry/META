% things_to_save['error_E'] = error_E
% things_to_save['error_V'] = error_V
% things_to_save['lambda_trace'] = lambda_trace

expectation_list = ["direct_greedy_results", ...
                    "off_togtd_00_results", ...
                    "off_togtd_02_results", ...
                    "off_togtd_04_results", ...
                    "off_togtd_06_results", ...
                    "off_togtd_08_results", ...
                    "off_togtd_10_results"];
LineColors = linspecer(numel(expectation_list) - 1);
LineColors = [[1, 0, 0]; LineColors];
num_points = 201;

CURVES = []; LEGENDS = {};
figure();

for result_index = 1: numel(expectation_list)
	result_name  = expectation_list(result_index);
    eval(sprintf('results = %s;', result_name))
    [X, MEAN, INTERVAL] = get_statistics(results, num_points);
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "direct_greedy_results")
        LEGEND = "greedy";
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
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 12);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
drawnow;