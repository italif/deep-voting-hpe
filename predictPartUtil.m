function [UE_part,total_part,geom_part,PI_part] = predictPartUtil(toPredict,pairs_part,N_trip,UE,total_binary_from_net,geometric_binary,oldPred,old_padd_size)
sz_coarse = [44,44];
C = get_C();
N_pts = C.N_pts;
sz = sqrt(numel(UE{1}));
N_pairs = size(pairs_part,1);
temp = l2(pairs_part,C.CRF_pairs_ex4_r(:,1:2));
[temp,pair_ind] = min(temp,[],2);
assert(max(abs(temp)) == 0)
%%
if ~exist('old_padd_size','var')
    old_padd_size = 1;
end
if ~exist('oldPred','var')
    oldPred = -1*ones(N_pts,1);
end
se = strel('disk',old_padd_size+1,0);
oldPredIdx = find(oldPred>0);
assert(~any(any(ismember(oldPredIdx,toPredict))))
assert(max(toPredict)<=N_pts)
assert(all(all(ismember(pairs_part+1,[toPredict(:); oldPredIdx(:)]) | pairs_part+1>N_pts)))
%% add UE for old points
for i = 1:numel(oldPredIdx)
    [r,c] =  ind2sub([sz,sz],oldPred(oldPredIdx(i)));
    temp = false([sz,sz]);
    temp(r,c) = true; 
    temp = ~imdilate(temp, se);
    UE{oldPredIdx(i)}(temp(:)) = 1e10; clear temp;
%     r_min = max(r-old_padd_size,1);
%     c_min = max(c-old_padd_size,1);
%     r_max = min(r+old_padd_size,sz);
%     c_max = min(c+old_padd_size,sz);
%     [r_tot,c_tot] = meshgrid(r_min:r_max,c_min:c_max);
%     r_tot = r_tot(:);
%     c_tot = c_tot(:);
%     ind = sub2ind([sz,sz],r_tot,c_tot);
%     temp = 1e10*ones(size(UE{oldPredIdx(i)}));
%     temp(ind) = UE{oldPredIdx(i)}(ind);
%     UE{oldPredIdx(i)} = temp;
end
%% create UE for new points
UE_part = cell(N_pts,1);
for i = 1:N_pts
    UE_part{i} = -1;
end
for i = 1:numel(toPredict)
    UE_part{toPredict(i)} = UE{toPredict(i)};
end
for i = 1:numel(oldPredIdx)
    UE_part{oldPredIdx(i)} = UE{oldPredIdx(i)};
end
%% create new edges
total_part = cell(N_pairs-2*N_trip,1);
geom_part = cell(N_pairs-2*N_trip,1);
PI_part = uint32(zeros(3,N_pairs-2*N_trip));

for i = 1:N_trip
    point1 = pairs_part(3*i,1)+1;
    point2 = pairs_part(3*i,2)+1;
    idx1 = 3*i-2; idx2 = 3*i-1;
    assert(pairs_part(idx1,2) == pairs_part(idx2,1))
    mid_point = pairs_part(idx1,2);
    
    bin1 = reshape(total_binary_from_net{pair_ind(idx1)},[sz_coarse,sz_coarse]);
    bin2 = reshape(total_binary_from_net{pair_ind(idx2)},[sz_coarse,sz_coarse]);
    unar = reshape(UE{mid_point+1},sz_coarse);
    
    
    [r1,c1,r2,c2] = ind2sub([sz_coarse,sz_coarse],1:prod([sz_coarse,sz_coarse]));
    r_mid = round(mean([r1(:),r2(:)],2));
    c_mid = round(mean([c1(:),c2(:)],2));
    locMid = sub2ind(sz_coarse,r_mid,c_mid);
    
    loc_bin_1 = sub2ind([sz_coarse,sz_coarse],r1,c1,r_mid',c_mid');
    loc_bin_2 = sub2ind([sz_coarse,sz_coarse],r_mid',c_mid',r2,c2);
    
    temp = 0.5*(bin1(loc_bin_1)+bin2(loc_bin_2))+unar(locMid');
    temp = reshape(temp,[prod(sz_coarse),prod(sz_coarse)]);
    
    total_part{i} = temp;%+total_binary_from_net{3*i};
    try
        geom_part{i} = geometric_binary{pair_ind(3*i)};
    catch
        geom_part{i} = [];
    end
    
    PI_part(:,i) = [point1; point2;i];
end
%%
offset = 2*N_trip;
for i = 3*N_trip+1:N_pairs
    point1 = pairs_part(i,1)+1;
    point2 = pairs_part(i,2)+1;

    
    total_part{i-offset} = total_binary_from_net{pair_ind(i)};
    try
        geom_part{i-offset} =  geometric_binary{pair_ind(i)};
    catch
        geom_part{i-offset} = [];
    end
    PI_part(:,i-offset) = [point1; point2;i-offset];
end
end

function D = l2(A,B)
% calcualtes the squared L2 distance
% A - NxD
% B - MxD
% D - NxM
D = bsxfun(@plus, sum(B.^2,2)', bsxfun(@plus, sum(A.^2,2), -2*A*B'));
end
