% things_to_save['error_L_exp'] = error_L_exp
% things_to_save['error_L_var'] = error_L_var

expectation_list = ["lambda_mta", ...
                    "lambda_greedy"];
LineColors = linspecer(numel(expectation_list) - 2);
LineColors = [[0, 0, 1]; LineColors; [1, 0, 0]];
num_points = 201;

% filename = 'mta_N_11_behavior_0.15_target_0.05_episodes_5000.mat';
% load(filename);

CURVES = []; LEGENDS = {};
figure();

for result_index = 1: numel(expectation_list)
	result_name  = expectation_list(result_index);
    eval(sprintf('results = %s;', result_name))
    [X, MEAN, INTERVAL] = get_statistics(results, num_points, true);
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL, LineColors(result_index, :)); %X, MEAN, INTERVAL, COLOR
    CURVES = [CURVES, CURVE];
    if strcmp(result_name, "lambda_mta")
        LEGEND = "MTA";
    elseif strcmp(result_name, "lambda_greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(L, 'Location', 'east');
set(gca, 'xscale', 'log');
axis([0, inf, 0, 1]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;