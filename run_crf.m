function predictions = run_crf(new_annorect, geometric_binary, C, single_person, num_iter, debugFlag)
if ~exist('debugFlag','var')
    debugFlag = 0;
end

if ~exist('num_iter','var')
    num_iter = 15;% num trw-s iters
end
predictions = [];

%% params
useGeo = 1;
min_weight = 1e-30; % min unary weights
dist_bound = 1;
use_restricted_mid = 1;
if (single_person)
    se4 = strel('disk',4,0);
else
    se4 = strel('disk',1,0);
end

pairs = C.CRF_pairs_ex4_r;
N_points_dense = C.N_dense_pts;
load('./dists_maps/offset_b_given_a_small_res12.mat');

%% part1 points
toPredict1 = [C.pelvis,C.thorax,C.upper_neck,C.head_top]+1;
pairs_part1 = [...
    C.head_top, C.head_center;
    C.head_center, C.upper_neck;
    C.head_top, C.upper_neck
    C.thorax, C.mid_body;
    C.mid_body, C.pelvis;
    C.thorax, C.pelvis;
    C.upper_neck, C.thorax;
    ];
N_trip1 = 2;
%%
toPredict2 = [C.head_top, C.upper_neck, C.l_shoulder, C.r_shoulder,C.l_hip, C.r_hip]+1;
pairs_part2 = [...
    C.head_top, C.head_center;
    C.head_center, C.upper_neck;
    C.head_top, C.upper_neck;
    %
    C.r_shoulder, C.r_body;
    C.r_body, C.r_hip;
    C.r_shoulder, C.r_hip;
    
    C.l_shoulder, C.l_body;
    C.l_body, C.l_hip;
    C.l_shoulder, C.l_hip
    
    C.r_shoulder, C.thorax;
    C.thorax, C.l_shoulder;
    C.r_shoulder, C.l_shoulder;
    
    C.r_hip, C.pelvis;
    C.pelvis, C.l_hip;
    C.r_hip, C.l_hip;
    
    C.upper_neck, C.r_shoulder;
    C.upper_neck, C.l_shoulder;
    C.upper_neck, C.thorax];
N_trip2 = 5;
%%
pairs_part3 = [...
    C.head_top, C.head_center;
    C.head_center, C.upper_neck;
    C.head_top, C.upper_neck;
    
    C.r_shoulder, C.r_up_arm;
    C.r_up_arm, C.r_elbow;
    C.r_shoulder, C.r_elbow;
    
    C.r_elbow, C.r_low_arm;
    C.r_low_arm, C.r_wrist;
    C.r_elbow, C.r_wrist;
    
    
    C.l_shoulder, C.l_up_arm;
    C.l_up_arm, C.l_elbow;
    C.l_shoulder, C.l_elbow;
    
    C.l_elbow, C.l_low_arm;
    C.l_low_arm, C.l_wrist;
    C.l_elbow, C.l_wrist;
    
    C.thorax, C.mid_body;
    C.mid_body, C.pelvis;
    C.thorax, C.pelvis;
    
    
    C.r_hip, C.r_up_leg;
    C.r_up_leg, C.r_knee;
    C.r_hip, C.r_knee;
    
    C.r_knee, C.r_low_leg;
    C.r_low_leg, C.r_ankle;
    C.r_knee, C.r_ankle;
    
    C.l_hip, C.l_up_leg;
    C.l_up_leg, C.l_knee;
    C.l_hip, C.l_knee;
    
    C.l_knee, C.l_low_leg;
    C.l_low_leg, C.l_ankle;
    C.l_knee, C.l_ankle;
    
    C.r_shoulder, C.r_body;
    C.r_body, C.r_hip;
    C.r_shoulder, C.r_hip;
    
    C.l_shoulder, C.l_body;
    C.l_body, C.l_hip;
    C.l_shoulder, C.l_hip;
    
    C.r_shoulder, C.thorax;
    C.thorax, C.l_shoulder;
    C.r_shoulder, C.l_shoulder;
    
    C.r_hip, C.pelvis;
    C.pelvis, C.l_hip;
    C.r_hip, C.l_hip;
    
    C.upper_neck, C.r_shoulder;
    C.upper_neck, C.l_shoulder;
    C.upper_neck, C.thorax;
    
    C.r_wrist, C.r_hand;
    
    C.l_wrist, C.l_hand];
N_trip3 = 14;

[w1,w2] = meshgrid(logspace(-1,3,15),logspace(-1,3,15));
w1_part1 = w1(16);
w2_part1 = w2(16);

w1_part2 = w1(16);
w2_part2 = w2(16);

w1 = w1(95);
w2 = w2(95);

if (~useGeo)
    warning('No geometric binary');
    w2_part1 = 0;
    w2_part2 = 0;
    w2 = 0;
end

n_rects = length(new_annorect);
for i_rect=1:n_rects
    fprintf('run crf on person %d/%d\n', i_rect, n_rects);
    imsize = size(new_annorect(i_rect).image); imsize = imsize(1:2);
    
    pretrainedData = new_annorect(i_rect);
    sz_coarse = size(pretrainedData.CRF_pairs(1).binary_a_b);
    sz_coarse = sz_coarse(1:2);
    
    %% unary potentials
    UE = cell(N_points_dense,1);
    sz_unary = size(pretrainedData.probs.unary{1});
    sz_coarse_unary = size(pretrainedData.probs.coarse_unary{1});
    d = (sz_coarse_unary-sz_coarse)/2;
    for ic = 1:N_points_dense
        heat = pretrainedData.probs.coarse_unary{ic}(d+1:end-d,d+1:end-d);
        UE{ic} = -log(double(heat(:))+min_weight)';
    end; clear heat;
    UE2 = UE;
    
    %% force mid_body to be in center
    if use_restricted_mid
        d = (sz_coarse_unary-sz_coarse)/2;
        heat = pretrainedData.mid_prob.coarse_unary(d+1:end-d,d+1:end-d);
        UE2{C.mid_body+1} = -log(double(heat(:))+min_weight)';
        [~,mid_loc] = max(heat(:));
        [r_loc,c_loc] = ind2sub(size(heat),mid_loc);
        temp = false(sz_coarse);
        temp(r_loc,c_loc) = true;
        temp = ~imdilate(temp, se4);
        UE2{C.mid_body+1}(temp(:)) = 1e10; clear temp;
    end
    %%
    CRF_pairs = pretrainedData.CRF_pairs;
    offset_d = sqrt(sum(offset_b_given_a.^2,2));
    coarse_scale = 12;
    N_pairs_dense = size(pairs,1);
    [x,y] = meshgrid(1:sz_coarse(1),1:sz_coarse(2));
    x = x(:); y = y(:);
    total_binary_from_net = cell(N_pairs_dense,1);
    for pair = 1:N_pairs_dense
        temp = zeros(prod(sz_coarse),prod(sz_coarse));
        binary_a_b = CRF_pairs(pair).binary_a_b;
        if (~isempty(binary_a_b))
            for i = 1:numel(x)
                b = squeeze(binary_a_b(x(i),y(i),:));
                location = bsxfun(@plus,offset_b_given_a,[x(i),y(i)]);
                valid = location(:,1)>0 & location(:,2)>0 & location(:,1)<= sz_coarse(1) & location(:,2)<= sz_coarse(2) ;
                if dist_bound
                    valid = valid & (offset_d<=pairs(pair,3)/coarse_scale);
                end
                b = b(valid);
                location = location(valid,:);
                b_idx = sub2ind(sz_coarse,location(:,1),location(:,2));
                a_idx = sub2ind(sz_coarse,x(i),y(i));
                temp(a_idx,b_idx(:)) = b;
            end
            total_binary_from_net{pair} = -log(temp+min_weight);
        end
    end
    
    %% part1
    disp('part 1');
    [UE_part,total_part,geom_part,PI_part] = predictPartUtil(toPredict1,pairs_part1,N_trip1,UE2,total_binary_from_net,geometric_binary);
    
    if (useGeo)
        PE_part = cellfun(@(x,y) x*w1_part1-w2_part1*log(y),total_part,geom_part,'UniformOutput',0);
    else
        PE_part = cellfun(@(x) x*w1_part1,total_part,'UniformOutput',0);
    end
    
    L = vgg_trw_bp(UE_part, PI_part, PE_part,int32([1,0,num_iter]));
    
    %% part 2
    disp('part 2');
    oldPred = -1*ones(size(L));
    oldPred([C.pelvis,C.thorax]+1) = L([C.pelvis,C.thorax]+1);
    [UE_part,total_part,geom_part,PI_part] = predictPartUtil(toPredict2,pairs_part2,N_trip2,UE2,total_binary_from_net,geometric_binary,oldPred,3);
    
    if (useGeo)
        PE_part = cellfun(@(x,y) x*w1_part2-w2_part2*log(y),total_part,geom_part,'UniformOutput',0);
    else
        PE_part = cellfun(@(x) x*w1_part2,total_part,'UniformOutput',0);
    end
    
    L = vgg_trw_bp(UE_part, PI_part, PE_part,int32([1,0,num_iter]));
    
    %% part 3
    disp('part 3');
    oldPred = -1*ones(size(L));
    oldPred(toPredict2) = L(toPredict2);
    toPredict = 1:18;
    toPredict(find(oldPred>0)) = [];
    [UE_part,total_part,geom_part,PI_part] = predictPartUtil(toPredict,pairs_part3,N_trip3,UE2,total_binary_from_net,geometric_binary,oldPred,2);
    
    
    if (useGeo)
        PE_part = cellfun(@(x,y) x*w1-w2*log(y),total_part,geom_part,'UniformOutput',0);
    else
        PE_part = cellfun(@(x) x*w1,total_part,'UniformOutput',0);
    end
    
    L = vgg_trw_bp(UE_part, PI_part, PE_part,int32([1,0,num_iter]));
    
    %%
    
    coarse_pred = zeros(numel(UE_part),2);
    fine_pred = zeros(numel(UE_part),2);
    sz_up_unary = size(pretrainedData.probs.up_unary{1});
    d = (sz_coarse_unary-sz_coarse)/2;
    for i_pt = 1:size(coarse_pred,1)
        ind = L(i_pt);
        [coarse_pred(i_pt,1),coarse_pred(i_pt,2)] = ind2sub(sz_coarse,ind);
        pt_coarse_loc = coarse_pred(i_pt,:) + d;
        indx_pt = pretrainedData.probs.arg_coarse_unary{i_pt}(pt_coarse_loc(1),pt_coarse_loc(2));
        [r_pt,c_pt]=ind2sub(sz_up_unary, 1+indx_pt);
        fine_pred(i_pt,:) = [r_pt,c_pt]-(sz_up_unary-imsize)./2;
    end
    
    pred = fine_pred;
    predictions{i_rect} = pred;
    
end



%%
if (debugFlag)
    
    figure; imagesc(new_annorect(i_rect).image); axis equal; axis off;
    
    pairs_to_view = C.Dense_map;
    hold on
    for i = 1:size(pairs_to_view,1)
        start = [pred(pairs_to_view(i,1)+1,2),pred(pairs_to_view(i,1)+1,1)];
        finish = [pred(pairs_to_view(i,2)+1,2),pred(pairs_to_view(i,2)+1,1)];
        plot([start(1), finish(1)],[start(2),finish(2)],'LineWidth',2, 'Color', 'r');
    end
    %     title(sprintf('w1 = %f, w2 = %f',w1,w2))
    labels = strrep(C.Pts_list(1:18), '_', '.');
    text(pred(:,2), pred(:,1), labels, 'Color', 'y', 'VerticalAlignment', 'bottom');
    title('prediction');
   
    showHeatMap = false;
    if (showHeatMap)
        img = new_annorect(i_rect).image;
        key_pt_names = {'l_elbow', 'l_shoulder'};
        for key_pt_name = key_pt_names
            pt_id = eval(['C.' key_pt_name{1}]);
            xxx = reshape(exp(-UE2{pt_id+1}),sz_coarse);
            pp=imresize(xxx./max(xxx(:)),12);
            d = (size(pp,1)-size(img,1))/2;
            pp = pp(d+1:end-d,d+1:end-d,:);
            rgb_mask = uint8(round(255*ind2rgb(round(255*pp./max(pp(:))), jet(256))));
            figure; imagesc(0.5*img+0.5*rgb_mask); axis equal; axis off;
            nn = key_pt_name{1}; nn(nn=='_')='.';
            title(nn);
        end
     end
end

end

