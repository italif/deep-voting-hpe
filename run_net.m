function new_annorects = run_net(new_annorects, C, single_person, debugFlag)
if ~exist('debugFlag','var')
    debugFlag = 0;
end

if (single_person)
    mid_vote_r = 100;
else
    mid_vote_r = 30;
end

exp_path = './net';

caffe.reset_all;
caffe.set_mode_gpu;

caffe_device.net = 0;
caffe_device.unary = 0;
caffe_device.binary = 0;
caffe_device.single = 0;

%%
% Initialize the network
disp('initialize networks');
tic
caffe.set_device(caffe_device.net);

model_def_file = [exp_path '/deploy.prototxt'];
model_file = [exp_path '/HPE-WIS.caffemodel'];
net_conv = caffe.Net(model_def_file, model_file, 'test'); % create net and load weights

net_deconv_model = [exp_path '/deploy_deconv.prototxt'];
net_deconv = caffe.Net(net_deconv_model, 'test');

net_binary_model = [exp_path '/deploy_binary.prototxt'];
net_binary = caffe.Net(net_binary_model, 'test');

net_single_deconv_model = [exp_path '/deploy_single_deconv.prototxt'];
net_single_deconv = caffe.Net(net_single_deconv_model, 'test');
toc

%% deconv weights
disp('prepare unary weights');
tic
load('./dists_maps/dists_class_49_coarsex4.mat');
label_map = dists_class+1;
label_map = label_map(end:-1:1,end:-1:1);
weights = accumarray(label_map(:), ones(size(label_map(:))));
weights = reshape([1./weights; 0],1,1,[]);
label_map = label_map(32:96,32:96,:);
label_weights = weights(label_map);
deconv_weights = zeros([size(label_weights,1),size(label_weights,2),1,length(weights)],'single');
for i_k=1:length(weights)-1
    deconv_weights(:,:,:,i_k)=label_weights.*(label_map==i_k);
end

%% prepare Unary
caffe.set_device(caffe_device.unary);
for i_pt=1:C.N_dense_pts
    layer_name = ['unary_' C.Pts_list{i_pt}];
    net_deconv.layers(layer_name).params(1).set_data(deconv_weights);
end

caffe.set_device(caffe_device.single);
layer_name = 'unary_pt';
net_single_deconv.layers(layer_name).params(1).set_data(deconv_weights);
toc

%% prepare Binary
disp('prepare binary weights');
caffe.set_device(caffe_device.binary);
tic
load('./dists_maps/dists_class_49_coarsex12.mat');
label_map_coarse = dists_class+1;
label_map_coarse = label_map_coarse(end:-1:1,end:-1:1);
weights_coarse = accumarray(label_map_coarse(:), ones(size(label_map_coarse(:))));
weights_coarse = reshape([1./weights_coarse; 0],1,1,[]);
label_map_coarse = label_map_coarse(17:27,17:27);
max_label = 12*2+1;
label_map_coarse(label_map_coarse>max_label)=0;

[x1,x2] = meshgrid(1:numel(label_map_coarse),1:numel(label_map_coarse)); % all combinations, not including non person
[r1,c1] = ind2sub(size(label_map_coarse),x1(:));
[r2,c2] = ind2sub(size(label_map_coarse),x2(:));
rc1 = [r1,c1];
rc2 = [r2,c2];
d_pt = (rc1-rc2);
[cc, ~, ic] = unique(d_pt,'rows');
n_channels = length(cc);
channel_sz = [size(label_map_coarse), 1, max_label^2];
coarse_channels = cell([1,1,n_channels]);
for i_cnl=1:n_channels
    rc1_c = rc1(ic==i_cnl,:);
    rc2_c = rc2(ic==i_cnl,:);
    ind1 = sub2ind(size(label_map_coarse),rc1_c(:,1),rc1_c(:,2));
    ind2 = sub2ind(size(label_map_coarse),rc2_c(:,1),rc2_c(:,2));
    bins1_c=label_map_coarse(ind1);
    bins2_c=label_map_coarse(ind2);
    rmov = (bins1_c==0 | bins2_c==0);
    bins1_c = bins1_c(~rmov);
    bins2_c = bins2_c(~rmov);
    rc1_c = rc1_c(~rmov,:);
    rc2_c = rc2_c(~rmov,:);
    chnl = (bins1_c-1).*max_label+bins2_c;
    coarse_channels{i_cnl}=zeros(channel_sz,'single');
    if (~isempty(chnl))
        coarse_channels{i_cnl}(sub2ind(channel_sz, rc1_c(:,1),rc1_c(:,2),chnl))=1; % 2 is conditioned on 1
    end
end
coarse_channels = cell2mat(coarse_channels);

net_binary.layers('binary_a_b').params(1).set_data(single(coarse_channels));
toc

%%
n_rects = length(new_annorects);
for i_rect=1:n_rects
    fprintf('run net on person %d/%d\n', i_rect, n_rects);
    img = new_annorects(i_rect).image;
    imsize = size(img); imsize = imsize(1:2);
    
    %% build unary
    tic
    disp('build unary');
    caffe.set_device(caffe_device.net);
    
    im = single(img);
    % mean BGR pixel
    mean_pix = [103.939, 116.779, 123.68];
    % RGB -> BGR
    im = im(:, :, [3 2 1]);
    % col major -> row major
    im = permute(im, [2 1 3]);
    % mean BGR pixel subtraction
    im = bsxfun(@minus,im,reshape(mean_pix,1,1,[]));
    net_conv.blobs('image').reshape([size(im) 1]);
    net_conv.blobs('image').set_data(im);
    net_conv.forward_prefilled();
    
    mid_pt_id = C.mid_body;
    mid_seed_pt_xy = (imsize+1)./2;

    i_pt=mid_pt_id+1;
    data_name = ['soft_' C.Pts_list{i_pt}];
    cmd = ['data = net_conv.blobs(''' data_name ''').get_data();'];
    eval(cmd);
    data = permute(data, [2 1 3]);
    
    dtsize = size(data); dtsize = dtsize(1:2);
    d = (imsize./4-dtsize)./2;
    r = mid_vote_r/4;
    p = mid_seed_pt_xy./4-d;
    [x,y]=meshgrid(1:dtsize(1),1:dtsize(2));
    pt_vote_mask = single(((x-p(1)).^2 + (y-p(2)).^2)<=r^2);
    data = bsxfun(@times, data, pt_vote_mask);
    
    caffe.set_device(caffe_device.single);
    net_single_deconv.blobs('soft_pt').reshape([size(data) 1]);
    net_single_deconv.blobs('soft_pt').set_data(data);
    net_single_deconv.forward_prefilled();
    
    mid_prob.unary = net_single_deconv.blobs('unary_pt').get_data();
    mid_prob.up_unary = net_single_deconv.blobs('up_unary_pt').get_data();
    mid_prob.coarse_unary = net_single_deconv.blobs('coarse_prob_pt').get_data();
    mid_prob.arg_coarse_unary = net_single_deconv.blobs('coarse_arg_pt').get_data();
    
    caffe.set_device(caffe_device.unary);
    clear coarse_soft;
    clear prob_non_person;
    clear coarse_prob_non_person;
    first_time = true;
    for i_pt=1:C.N_dense_pts
        data_name = ['soft_' C.Pts_list{i_pt}];
        cmd = ['data = net_conv.blobs(''' data_name ''').get_data();'];
        eval(cmd);
        data = permute(data, [2 1 3]);
        net_deconv.blobs(data_name).reshape([size(data) 1]);
        net_deconv.blobs(data_name).set_data(data);
        cmd = ['coarse_data = net_conv.blobs(''coarse_soft_' C.Pts_list{i_pt} ''').get_data();'];
        eval(cmd);
        coarse_data = permute(coarse_data, [2 1 3]);
        cmd = ['coarse_soft.prob_' C.Pts_list{i_pt} ' = coarse_data;'];
        eval(cmd);
        if first_time
            first_time = false;
            prob_non_person = data(:,:,end);
            coarse_prob_non_person = coarse_data(:,:,end);
        else
            prob_non_person = prob_non_person+data(:,:,end);
            coarse_prob_non_person = coarse_prob_non_person+coarse_data(:,:,end);
        end
    end
    net_deconv.forward_prefilled();
    
    clear data;
    clear probs;
    for i_pt=1:C.N_dense_pts
        cmd = ['probs.unary{' num2str(i_pt) '} = net_deconv.blobs(''unary_' C.Pts_list{i_pt} ''').get_data();'];
        eval(cmd);
        cmd = ['probs.up_unary{' num2str(i_pt) '} = net_deconv.blobs(''up_unary_' C.Pts_list{i_pt} ''').get_data();'];
        eval(cmd);
        cmd = ['probs.coarse_unary{' num2str(i_pt) '} = net_deconv.blobs(''coarse_prob_' C.Pts_list{i_pt} ''').get_data();'];
        eval(cmd);
        cmd = ['probs.arg_coarse_unary{' num2str(i_pt) '} = net_deconv.blobs(''coarse_arg_' C.Pts_list{i_pt} ''').get_data();'];
        eval(cmd);
    end
    probs.prob_non_person = prob_non_person;
    probs.coarse_prob_non_person = coarse_prob_non_person;
    
    CRF_pairs=[];
    build_CRF_pairs = true;
    if (build_CRF_pairs)
        %% build binary
        disp('build binary')
        caffe.set_device(caffe_device.binary);
        
        clear CRF_pairs;
        for i_pair = 1:length(C.CRF_pairs_ex4_r_b)
            binary_a_b = [];
            key_pt_name_a = C.Pts_indx_list{1+C.CRF_pairs_ex4_r_b(i_pair,1),2};%'r_shoulder';%'r_elbow';%'head_top';%'r_shoulder';%'head_center';
            key_pt_name_b = C.Pts_indx_list{1+C.CRF_pairs_ex4_r_b(i_pair,2),2};%'l_shoulder';%'r_wrist';%'thorax';%'r_up_arm';%'thorax';
            disp(['bulding binary for (' key_pt_name_a ',' key_pt_name_b ')']);
            
            if (C.CRF_pairs_ex4_r_b(i_pair,4))
                eval(['a = bsxfun(@times,coarse_soft.prob_' key_pt_name_a ', reshape(weights_coarse,1,1,[]));']);
                eval(['b = bsxfun(@times,coarse_soft.prob_' key_pt_name_b ', reshape(weights_coarse,1,1,[]));']);
                [ax,bx]=meshgrid(1:max_label,1:max_label);
                ax = ax(:); ax=reshape(ax,1,1,[]); ax=repmat(ax,[size(a,1),size(a,2),1]);
                bx = bx(:); bx=reshape(bx,1,1,[]); bx=repmat(bx,[size(b,1),size(b,2),1]);
                [r,c]=meshgrid(1:size(ax,1),1:size(ax,2));
                r=repmat(r,[1,1,size(ax,3)]);
                c=repmat(c,[1,1,size(ax,3)]);
                ax = sub2ind(size(a),c,r,ax);
                bx = sub2ind(size(b),c,r,bx);
                data = a(ax).*b(bx);
                
                net_binary.blobs('coarse_soft_a_b').reshape([size(data) 1]);
                net_binary.blobs('coarse_soft_a_b').set_data(single(data));
                
                net_binary.forward_prefilled();
                
                binary_a_b = net_binary.blobs('binary_a_b').get_data();
            end
            CRF_pairs(i_pair).key_pt_name_a = key_pt_name_a;
            CRF_pairs(i_pair).key_pt_name_b = key_pt_name_b;
            CRF_pairs(i_pair).binary_a_b = binary_a_b;
        end
        
        toc
    end
    iii=0;
    
    if (debugFlag)
        %%
        coarse_a = sum(binary_a_b,3);
        [max_r,max_c]=find(coarse_a==max(coarse_a(:)));
        r = max_r(1);
        c = max_c(1);
        b_given_a = binary_a_b(r,c,:);
        sz = 2*size(label_map_coarse)-1;
        b_given_a_map = zeros(sz);
        rc = bsxfun(@plus,-cc,size(label_map_coarse));
        indx = sub2ind(sz,rc(:,1),rc(:,2));
        b_given_a_map(indx)=b_given_a;
        d = (sz-1)./2;
        conditional_map = zeros(size(coarse_a));
        conditional_map(r-d:r+d,c-d:c+d) = b_given_a_map;
        
        %figure; imagesc(coarse_a); axis equal; axis off;
        xxx = coarse_a;
        pp=imresize(xxx./max(xxx(:)),12);
        d = (size(pp,1)-size(img,1))/2;
        pp = pp(d+1:end-d,d+1:end-d,:);
        rgb_mask = uint8(round(255*ind2rgb(round(255*pp./max(pp(:))), jet(256))));
        figure; imagesc(0.5*img+0.5*rgb_mask); axis equal; axis off;
        nn_a = key_pt_name_a; nn_a(nn_a=='_')='.';
        title(['coarse ' nn_a]);
        
        eval(['temp=(probs.up_unary{C.' key_pt_name_a '+1});']);
        %figure; imagesc(temp); axis equal; axis off;
        xxx = temp;
        pp = xxx./max(xxx(:));
        d = (size(pp,1)-size(img,1))/2;
        pp = pp(d+1:end-d,d+1:end-d,:);
        rgb_mask = uint8(round(255*ind2rgb(round(255*pp./max(pp(:))), jet(256))));
        figure; imagesc(0.5*img+0.5*rgb_mask); axis equal; axis off;
        nn_a = key_pt_name_a; nn_a(nn_a=='_')='.';
        title(['fine ' nn_a]);
        
        %figure; imagesc(conditional_map); axis equal; axis off;
        xxx = conditional_map;
        pp=imresize(xxx./max(xxx(:)),12);
        d = (size(pp,1)-size(img,1))/2;
        pp = pp(d+1:end-d,d+1:end-d,:);
        rgb_mask = uint8(round(255*ind2rgb(round(255*pp./max(pp(:))), jet(256))));
        figure; imagesc(0.5*img+0.5*rgb_mask); axis equal; axis off;
        nn_a = key_pt_name_a; nn_a(nn_a=='_')='.';
        nn_b = key_pt_name_b; nn_b(nn_b=='_')='.';
        title(['best ' nn_b ' location given ' nn_a ' with max location']);
        
        eval(['temp = (probs.up_unary{C.' key_pt_name_b '+1});']);
        %figure; imagesc(temp); axis equal; axis off;
        xxx = temp;
        pp = xxx./max(xxx(:));
        d = (size(pp,1)-size(img,1))/2;
        pp = pp(d+1:end-d,d+1:end-d,:);
        rgb_mask = uint8(round(255*ind2rgb(round(255*pp./max(pp(:))), jet(256))));
        figure; imagesc(0.5*img+0.5*rgb_mask); axis equal; axis off;
        nn_b = key_pt_name_b; nn_b(nn_b=='_')='.';
        title(['fine ' nn_b]);
    end
    new_annorects(i_rect).CRF_pairs = CRF_pairs;
    new_annorects(i_rect).probs = probs;
    new_annorects(i_rect).mid_prob = mid_prob;
end

caffe.reset_all;

end
