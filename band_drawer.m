function [CURVE, BAND] = band_drawer(varargin)
set(gcf, 'Renderer', 'Painter');
if nargin == 3 % X, MEAN, INTERVAL
    X = varargin{1};
    MEAN = varargin{2};
    INTERVAL = varargin{3};
    LineWidth = 1;
    Curve = plot(X, MEAN, 'LineWidth', LineWidth);
elseif nargin == 4 % X, MEAN, INTERVAL, COLOR
    X = varargin{1};
    MEAN = varargin{2};
    INTERVAL = varargin{3};
    COLOR = varargin{4};
    if norm(COLOR - [0, 0, 1]) == 0
        LineWidth = 2;
    elseif norm(COLOR - [1, 0, 0]) == 0
        LineWidth = 2;
    else
        LineWidth = 1;
    end
    Curve = plot(X, MEAN, 'LineWidth', LineWidth, 'COLOR', COLOR);
end
COLOR = Curve.Color;
delete(Curve);
BAND = fill([X, flip(X)], [INTERVAL(1, :), flip(INTERVAL(2, :))], COLOR, 'EdgeColor', 'none', 'FaceAlpha', '0.4');
hold on;
CURVE = plot(X, MEAN, '-', 'LineWidth', LineWidth, 'COLOR', COLOR);
drawnow;
end