function str_legend = get_legend(str_method)
if strcmp(str_method, "togtd_0") || strcmp(str_method, "baseline_0")
    str_legend = "gtd(0)";
elseif strcmp(str_method, "togtd_20") || strcmp(str_method, "baseline_20")
    str_legend = "gtd(.2)";
elseif strcmp(str_method, "togtd_40") || strcmp(str_method, "baseline_40") || strcmp(str_method, "togtd_400") || strcmp(str_method, "baseline_400")
    str_legend = "gtd(.4)";
elseif strcmp(str_method, "togtd_60") || strcmp(str_method, "baseline_60")
    str_legend = "gtd(.6)";
elseif strcmp(str_method, "togtd_80") || strcmp(str_method, "togtd_800") || strcmp(str_method, "baseline_80") || strcmp(str_method, "baseline_800")
    str_legend = "gtd(.8)";
elseif strcmp(str_method, "togtd_90") || strcmp(str_method, "togtd_900") || strcmp(str_method, "baseline_90") || strcmp(str_method, "baseline_900")
    str_legend = "gtd(.9)";
elseif strcmp(str_method, "togtd_950") || strcmp(str_method, "baseline_950")
    str_legend = "gtd(.95)";
elseif strcmp(str_method, "togtd_975") || strcmp(str_method, "baseline_975")
    str_legend = "gtd(.975)";
elseif strcmp(str_method, "togtd_990") || strcmp(str_method, "baseline_990")
    str_legend = "gtd(.99)";
elseif strcmp(str_method, "togtd_100") || strcmp(str_method, "togtd_1000") || strcmp(str_method, "baseline_100") || strcmp(str_method, "baseline_1000")
    str_legend = "gtd(1)";
elseif strcmp(str_method, "totd_0")
    str_legend = "td(0)";
elseif strcmp(str_method, "totd_20")
    str_legend = "td(.2)";
elseif strcmp(str_method, "totd_40") || strcmp(str_method, "totd_400")
    str_legend = "td(.4)";
elseif strcmp(str_method, "totd_60")
    str_legend = "td(.6)";
elseif strcmp(str_method, "totd_80") || strcmp(str_method, "totd_800")
    str_legend = "td(.8)";
elseif strcmp(str_method, "totd_90") || strcmp(str_method, "totd_900")
    str_legend = "td(.9)";
elseif strcmp(str_method, "totd_950")
    str_legend = "td(.95)";
elseif strcmp(str_method, "totd_975")
    str_legend = "td(.975)";
elseif strcmp(str_method, "totd_990")
    str_legend = "td(.99)";
elseif strcmp(str_method, "totd_100") || strcmp(str_method, "totd_1000")
    str_legend = "td(1)";
elseif strcmp(str_method, "greedy")
    str_legend = "greedy";
elseif strcmp(str_method, "mta_nonparam") || strcmp(str_method, "MTA_nonparam")
    str_legend = "M*(np)";
elseif strcmp(str_method, "mta") || strcmp(str_method, "MTA")
    str_legend = "M*";
end
end

