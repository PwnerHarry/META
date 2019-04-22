% things_to_save['error_L_var'] = error_L_var
% things_to_save['error_var_greedy'] = error_var_greedy

expectation_list = ["error_L_var", ...
                    "error_var_greedy"];
LineColors = linspecer(numel(expectation_list) - 2);
LineColors = [[0, 0, 1]; LineColors; [1, 0, 0]];
num_points = 1001;

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
    if strcmp(result_name, "error_L_var")
        LEGEND = "MTA";
    elseif strcmp(result_name, "error_var_greedy")
        LEGEND = "greedy";
    end
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 12);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
drawnow;