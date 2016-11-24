function annorects = run_on_image(img, annorects, single_person, upper_only, geometric_binary)

if (~exist('single_person','var'))
    single_person = true;
end
if (~exist('upper_only','var'))
    upper_only = false;
end
if (~exist('geometric_binary','var'))
    disp('load pretrain data')
    geometric_binary = get_geometric_binary();
end

C = get_C();

new_annorects = get_new_annorect(img, annorects);
new_annorects = run_net(new_annorects, C, single_person);
if (upper_only)
    predictions = run_crf_upper(new_annorects, geometric_binary, C, single_person);
else
    predictions = run_crf(new_annorects, geometric_binary, C, single_person);
end
annorects = transform_back_pred(annorects, new_annorects, predictions);
end

function geometric_binary = get_geometric_binary();
load('dists_maps/geometric_binary_small_ex_4.mat','geometric_binary');
% clean geometric
min_weight = 1e-30;
for i = 1:numel(geometric_binary)
    geometric_binary{i}(abs(geometric_binary{i})<1e-10) = min_weight;
end
end