function [X, MEAN, INTERVAL] = get_statistics(results, num_points, log_flag)
if log_flag || isempty(log_flag)
episodes = size(results, 2);
ratio = nthroot(episodes, num_points - 1);
X = max(ones(1, num_points), round(ratio .^ (0: num_points - 1)));
else
    episodes = size(results, 2);
    X = max(ones(size(linspace(0, 1, num_points))), round(episodes * linspace(0, 1, num_points)));
end
results = results(:, X);
episodes = size(results, 2);
MEAN = mean(results, 1, 'omitnan');
INTERVAL = NaN(2, episodes);
for episode = 1: episodes
    D = results(:, episode);
    if sum(D) == 0
        INTERVAL(:, episode) = [0, 0]';
        continue;
    end
    D(D == inf) = [];
    D(isnan(D)) = [];
    D(D == 0) = eps;
    if length(D) <= 1
        INTERVAL(:, episode) = INTERVAL(:, episode - 1);
    else
        [phat, pci] = mle(D, 'dist', 'logn');
        INTERVAL(:, episode) = MEAN(episode) + 2 * (exp(pci(:, 1)) - exp(phat(1)));
        if INTERVAL(1, episode) <= 0
            INTERVAL(1, episode) = MEAN(episode) + (INTERVAL(1, episode - 1) - MEAN(episode - 1)) * MEAN(episode) / MEAN(episode - 1);
        end
        INTERVAL(:, episode) = max(INTERVAL(:, episode), 1e-14 * ones(size(INTERVAL(:, episode))));
    end
end
end