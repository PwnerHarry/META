folder = 'F:\MTA_DATA\ringworld\100000';
behavior = 0.15;
target = 0.05;
dirOutput = dir(fullfile(folder, '*'));
subfolders = {dirOutput.name}';
subfolders(1: 2) = [];
MTA_VALUES = {};
CURVES = [];
LEGENDS = {};
KAPPAS = [];
VALUES1 = [];
VALUES2 = [];
% load greedy baseline
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for j = 1: numel(filenames)
    if isempty(strfind(filenames{j}, ['behavior_', num2str(behavior)])) || isempty(strfind(filenames{j}, ['target_', num2str(target)]))
        reduce_index = [reduce_index, j];
    end
    if ~isempty(strfind(filenames{j}, 'kappa'))
        reduce_index = [reduce_index, j];
    end
end
filenames(reduce_index) = [];
loaded = load(fullfile(folder, filenames{1}));
[~, MEAN, ~] = get_statistics(loaded.error_value_greedy, 201, true);
BASELINE_GREEDY = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_0, 201, true);
BASELINE_TOGTD_0 = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_20000, 201, true);
BASELINE_TOGTD_20000 = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_40000, 201, true);
BASELINE_TOGTD_40000 = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_60000, 201, true);
BASELINE_TOGTD_60000 = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_80000, 201, true);
BASELINE_TOGTD_80000 = mean(MEAN(end - 20: end));
[~, MEAN, ~] = get_statistics(loaded.error_value_togtd_100000, 201, true);
BASELINE_TOGTD_100000 = mean(MEAN(end - 20: end));

for i = 1: length(subfolders)
    subfoler = subfolders{i};
    dirOutput = dir(fullfile(folder, subfoler, '*'));
    filenames = {dirOutput.name}';
    reduce_index = [];
    for j = 1: numel(filenames)
        if isempty(strfind(filenames{j}, ['behavior_', num2str(behavior)])) || isempty(strfind(filenames{j}, ['target_', num2str(target)]))
            reduce_index = [reduce_index, j];
        end
    end
    filenames(reduce_index) = [];
    try
        filename = filenames{1};
        loaded = load(fullfile(folder, subfoler, filename));
    catch ME
        continue;
    end
    
    [X, MEAN, INTERVAL] = get_statistics(loaded.error_value_mta, 201, true);
    KAPPAS = [KAPPAS, str2double(subfoler)];
    VALUES2 = [VALUES2, mean(MEAN(end - 20 : end))];
    VALUES1 = [VALUES1, mean(MEAN(round(0.4 * length(MEAN) - 10) : round(0.4 * length(MEAN) + 10)))];
    % 	if strcmp(subfoler, '0.2') || strcmp(subfoler, '0.1')
    %         continue;
    %     end
    [CURVE, ~] = band_drawer(X, MEAN, INTERVAL);
    CURVES = [CURVES, CURVE];
    LEGEND = sprintf('\\kappa=%s', subfoler);
    LEGENDS = [LEGENDS, LEGEND];
end

L = legend(CURVES, LEGENDS{:});
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
set(L, 'Location', 'southwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([1, inf, 0, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
drawnow;

% figure;
% CURVE_MTA = plot(KAPPAS, VALUES1, '-o', 'LineWidth', 2);
% hold on;
% set(gca, 'xscale', 'log');
% set(gca, 'yscale', 'log');
% axis([0, inf, 0, inf]);
% set(gca, 'FontSize', 16);
% set(gca, 'FontName', 'Book Antiqua');
% drawnow;

MIN = min(KAPPAS);
MAX = max(KAPPAS);
figure;
CURVE_MTA = plot(KAPPAS, VALUES2, '-o', 'LineWidth', 2);
hold on;
CURVE_GREEDY = plot([MIN, MAX], BASELINE_GREEDY * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_0 = plot([MIN, MAX], BASELINE_TOGTD_0 * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_20000 = plot([MIN, MAX], BASELINE_TOGTD_20000 * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_40000 = plot([MIN, MAX], BASELINE_TOGTD_40000 * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_60000 = plot([MIN, MAX], BASELINE_TOGTD_60000 * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_80000 = plot([MIN, MAX], BASELINE_TOGTD_80000 * ones(2, 1), 'LineWidth', 2);
% CURVE_TOGTD_100000 = plot([MIN, MAX], BASELINE_TOGTD_100000 * ones(2, 1), 'LineWidth', 2);
% L = legend([CURVE_MTA, CURVE_GREEDY, CURVE_TOGTD_0, CURVE_TOGTD_20000, CURVE_TOGTD_40000, CURVE_TOGTD_60000, CURVE_TOGTD_80000, CURVE_TOGTD_100000], ...
%     {'MTA', 'GREEDY', 'GTD(0)', 'GTD(0.2)', 'GTD(0.4)', 'GTD(0.6)', 'GTD(0.8)', 'GTD(1)'});
L = legend([CURVE_MTA, CURVE_GREEDY], {'MTA', 'GREEDY'});
axis([0, inf, 0, inf]);
set(gca, 'FontSize', 16);
set(gca, 'FontName', 'Book Antiqua');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
drawnow;