function [ALPHAS_UNI, LAMBDAS_UNI, KAPPAS_UNI, table_baseline, table_greedy, table_META, table_META_np] = get_summary(env, folder, pattern)
filenames = get_filenames(folder, pattern);
if strcmp(env, 'ringworld')
    METHOD_LIST = {'totd_0', 'totd_400', 'totd_800', 'totd_900', 'totd_950', 'totd_975', 'totd_990', 'totd_1000', 'greedy', 'mta'};
    smoothing_window = 100;
elseif strcmp(env, 'frozenlake')
    METHOD_LIST = {'togtd_0', 'togtd_400', 'togtd_800', 'togtd_900', 'togtd_950', 'togtd_975', 'togtd_990', 'togtd_1000', 'greedy', 'mta_nonparam', 'mta'};
    smoothing_window = 25;
elseif strcmp(env, 'mountaincar')
    METHOD_LIST = {'baseline_0', 'baseline_400', 'baseline_800', 'baseline_900', 'baseline_950', 'baseline_975', 'baseline_1000', 'greedy', 'MTA'}; % 'baseline_1000',
    smoothing_window = 100;
end
MEANS = nan(numel(METHOD_LIST), numel(filenames));
STDS = nan(numel(METHOD_LIST), numel(filenames));
ALPHAS = zeros(numel(filenames), 1);
KAPPAS = zeros(numel(filenames), 1);
for index_filename = 1: numel(filenames)
    filename = filenames{index_filename};
    if strcmp(env, 'ringworld')
        [index_start_alpha, index_end_alpha] = regexp(filename, 'a\_.*\_k');
        if isempty(index_start_alpha) && isempty(index_end_alpha)
            [index_start_alpha, index_end_alpha] = regexp(filename, 'a\_.*\_e');
        end
    elseif strcmp(env, 'frozenlake')
        [index_start_alpha, index_end_alpha] = regexp(filename, 'a\_.*\_b');
    elseif strcmp(env, 'mountaincar')
        [index_start_alpha, index_end_alpha] = regexp(filename, 'a\_.*\_b');
    end
    [index_start_kappa, index_end_kappa] = regexp(filename, 'k\_.*\_e');
    if isempty(index_start_kappa) && isempty(index_end_kappa)
        KAPPAS(index_filename) = 0; % baseline and greedy
    else
        KAPPAS(index_filename) = str2double(filename(index_start_kappa + 2: index_end_kappa - 2)); % META
    end
    ALPHAS(index_filename) = str2double(filename(index_start_alpha + 2: index_end_alpha - 2));
    loaded = load(fullfile(folder, filename)); %#ok<NASGU>
    for index_method = 1: numel(METHOD_LIST)
        method = METHOD_LIST{index_method};
        try
            if strcmp(env, 'mountaincar')
                eval(sprintf('index_not_nan = find(~isnan(loaded.return_%s_mean));', method));
                index_end = index_not_nan(end);
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.return_%s_mean(index_end - %d: index_end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.return_%s_std(index_end - %d: index_end), ''omitnan'');', index_method, method, smoothing_window));
            else
                eval(sprintf('MEANS(%d, index_filename) = mean(loaded.error_value_%s_mean(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
                eval(sprintf('STDS(%d, index_filename) = mean(loaded.error_value_%s_std(end - %d: end), ''omitnan'');', index_method, method, smoothing_window));
            end
        catch ME
        end
    end
end
ALPHAS_UNI = unique(ALPHAS);
KAPPAS_UNI = setdiff(unique(KAPPAS), zeros(1, 1));

% summarize table for META
table_META = NaN(length(ALPHAS_UNI), length(KAPPAS_UNI), 2); % 1st channel for means and 2nd for stds
for index_alpha_uni = 1: length(ALPHAS_UNI)
    alpha = ALPHAS_UNI(index_alpha_uni);
    index_alphas = find(ALPHAS == alpha);
    for index_kappa_uni = 1: length(KAPPAS_UNI)
        kappa = KAPPAS_UNI(index_kappa_uni);
        index_kappas = find(KAPPAS == kappa);
        index = intersect(index_alphas, index_kappas);
        if length(index) > 1
            index_reduce = [];
            for i = 1: length(index)
                if isnan(MEANS(end, index(i)))
                    index_reduce = [index_reduce, i];
                end
            end
            index(index_reduce) = [];
        end
        table_META(index_alpha_uni, index_kappa_uni, 1) = MEANS(end, index);
        table_META(index_alpha_uni, index_kappa_uni, 2) = STDS(end, index);
    end
end

% summarize table for META_np
table_META_np = NaN(length(ALPHAS_UNI), length(KAPPAS_UNI), 2); % 1st channel for means and 2nd for stds
if strcmp(env, 'frozenlake')
    for index_alpha_uni = 1: length(ALPHAS_UNI)
        alpha = ALPHAS_UNI(index_alpha_uni);
        index_alphas = find(ALPHAS == alpha);
        for index_kappa_uni = 1: length(KAPPAS_UNI)
            kappa = KAPPAS_UNI(index_kappa_uni);
            index_kappas = find(KAPPAS == kappa);
            index = intersect(index_alphas, index_kappas);
            if length(index) > 1
                index_reduce = [];
                for i = 1: length(index)
                    if isnan(MEANS(end - 1, index(i)))
                        index_reduce = [index_reduce, i];
                    end
                end
                index(index_reduce) = [];
            end
            table_META_np(index_alpha_uni, index_kappa_uni, 1) = MEANS(end - 1, index);
            table_META_np(index_alpha_uni, index_kappa_uni, 2) = STDS(end - 1, index);
        end
    end
end

% summarize table for baseline
LAMBDAS_UNI = [];
list_baselines = METHOD_LIST;
for index_baseline = 1: length(list_baselines)
    baseline = list_baselines{index_baseline};
    matched = regexp(baseline, '\d+\.?\d*', 'match');
    if isempty(matched)
        continue;
    end
    lambda = str2double(matched{1, 1}) / 1000;
    LAMBDAS_UNI = [LAMBDAS_UNI, lambda];
end

table_baseline = NaN(length(ALPHAS_UNI), length(METHOD_LIST) - 2, 2);
for index_alpha_uni = 1: length(ALPHAS_UNI)
    alpha = ALPHAS_UNI(index_alpha_uni);
    index_alphas = find(ALPHAS == alpha);
    index_baselines = find(KAPPAS == 0);
    index_alphas = intersect(index_alphas, index_baselines);
    for index_lambda_uni = 1: length(LAMBDAS_UNI)
        table_baseline(index_alpha_uni, index_lambda_uni, 1) = MEANS(index_lambda_uni, index_alphas);
        table_baseline(index_alpha_uni, index_lambda_uni, 2) = STDS(index_lambda_uni, index_alphas);
    end
end

table_greedy = NaN(length(ALPHAS_UNI), 1, 2);
for index_alpha_uni = 1: length(ALPHAS_UNI)
    alpha = ALPHAS_UNI(index_alpha_uni);
    index_alphas = find(ALPHAS == alpha);
    index_baselines = find(KAPPAS == 0);
    index_alphas = intersect(index_alphas, index_baselines);
    table_greedy(index_alpha_uni, 1, 1) = MEANS(end - 1, index_alphas);
    table_greedy(index_alpha_uni, 1, 2) = STDS(end - 1, index_alphas);
end
end

