clear all;
figure;
folder = 'frozenlake';
policy = 'off';
dirOutput = dir(fullfile(folder, '*'));
filenames = {dirOutput.name}';
reduce_index = [];
for i = 1: numel(filenames)
    if isempty(strfind(filenames{i}, 'error_value')) || isempty(strfind(filenames{i}, policy))
        reduce_index = [reduce_index, i];
    end
end
filenames(reduce_index) = [];
for i = 1: length(filenames)
    filename = filenames{i};
    shelled_sample = load(fullfile(folder, filename));
    samples(i) = shelled_sample.sample;
end
CURVES = [];
LEGENDS = {};
VALUES2 = [];
KAPPAS = [];
for i = 1: length(samples)
    if ~isempty(strfind(samples(i).name, 'mta'))% && ~isempty(strfind(samples(i).name, '0.5'))
        [CURVE, ~] = band_drawer(samples(i).X, samples(i).MEAN, samples(i).INTERVAL); %, [0, 0, 1]
        name = strrep(samples(i).name, 'error_value_', '');
        name = strrep(name, policy, '');
        name = strrep(name, 'mta', '');
        name = strrep(name, '_', '');
        KAPPAS = [KAPPAS, str2double(name)];
        VALUES2 = [VALUES2, mean(samples(i).MEAN(end - 20 : end))];
    elseif ~isempty(strfind(samples(i).name, 'greedy'))
        BASELINE_GREEDY = mean(samples(i).MEAN(end - 20 : end));
        continue;
    else
        continue;
    end
    CURVES = [CURVES, CURVE];
    LEGEND = samples(i).name;
    LEGEND = strrep(LEGEND, 'error_value_', '');
    LEGEND = strrep(LEGEND, policy, '');
    LEGEND = strrep(LEGEND, '_', ' ');
    LEGEND = strrep(LEGEND, 'mta', 'MTA');
    LEGEND = strrep(LEGEND, '  ', ' ');
    LEGEND = strrep(LEGEND, 'togtd 0', 'GTD(0)');
    LEGEND = strrep(LEGEND, 'togtd 20000', 'GTD(0.2)');
    LEGEND = strrep(LEGEND, 'togtd 40000', 'GTD(0.4)');
    LEGEND = strrep(LEGEND, 'togtd 60000', 'GTD(0.6)');
    LEGEND = strrep(LEGEND, 'togtd 80000', 'GTD(0.8)');
    LEGEND = strrep(LEGEND, 'togtd 100000', 'GTD(1)');
    LEGENDS = [LEGENDS, LEGEND];
end
L = legend(CURVES, LEGENDS);
set(L, 'FontName', 'Book Antiqua', 'FontSize', 14);
set(L, 'Location', 'northwest');
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
axis([1, inf, 0, inf]);
drawnow;


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
set(L, 'FontName', 'Book Antiqua', 'FontSize', 18);
drawnow;