function new_annorect = get_new_annorect(img, annorects, debugFlag)
if ~exist('debugFlag','var')
    debugFlag = 0;
end

HEAD_SIZE = 64;
RADIUS = 252;

fprintf('processing image\n');
new_annorect = [];
for i_rect = 1:length(annorects)
    new_annorect(i_rect).objpos = [];
    new_annorect(i_rect).scale = [];
    new_annorect(i_rect).tform = [];
    new_annorect(i_rect).image = [];
end

for i_rect = 1:length(annorects)
    if (isempty(i_rect) || ...
            ~isfield(annorects(i_rect), 'scale') || ...
            isempty(annorects(i_rect).scale) || ...
            ~isfield(annorects(i_rect), 'objpos') || ...
            isempty(annorects(i_rect).objpos))
        continue;
    end
    
    annorect = annorects(i_rect);
    
    tform = projective2d([1 0 0; 0 1 0; 0 0 annorect.scale/(0.03*HEAD_SIZE)]);
    [t_img, Rout] = imwarp(img, tform);
    tform = projective2d(tform.T-[0,0,0;0,0,0;Rout.XWorldLimits(1)*tform.T(3,3),Rout.YWorldLimits(1)*tform.T(3,3),0]);
    [objpos.x,objpos.y] = transformPointsForward(tform,annorect.objpos.x,annorect.objpos.y);
    scale = annorect.scale./tform.T(3,3);
    
    pos_xy = round([objpos.x, objpos.y]);
    
    
    from_bbox = [pos_xy-RADIUS+1,pos_xy+RADIUS];
    to_bbox = [1,1,2*RADIUS,2*RADIUS];
    
    % if bbox starts out side image then waive top-left part of bbox
    shift_xy = max(1-from_bbox(1:2), [0,0]);
    to_bbox(1:2) = to_bbox(1:2) + shift_xy;
    from_bbox(1:2) = from_bbox(1:2) + shift_xy;
    
    % if bbox ends out side image then waive bottom-right part of bbox
    imsize = size(t_img); imsize=imsize(1:2);
    shift_xy = max(from_bbox(3:4)-fliplr(imsize), [0,0]);
    to_bbox(3:4) = to_bbox(3:4) - shift_xy;
    from_bbox(3:4) = from_bbox(3:4) - shift_xy;
    
    patch = uint8(zeros(2*RADIUS,2*RADIUS,3));
    patch(to_bbox(2):to_bbox(4),to_bbox(1):to_bbox(3),:) = ...
        t_img(from_bbox(2):from_bbox(4),from_bbox(1):from_bbox(3),:);
    
    shift_xy = to_bbox(1:2)-from_bbox(1:2);
    
    pos_xy = pos_xy+shift_xy;
    tform = projective2d(tform.T+[0,0,0;0,0,0;shift_xy(1)*tform.T(3,3),shift_xy(2)*tform.T(3,3),0]);
    
    if (debugFlag)
        figure(2*i_rect-1);
        imagesc(img); axis equal; axis off;
        hold on; plot(annorect.objpos.x,annorect.objpos.y,'or'); hold off;
        [i_x,i_y] = transformPointsInverse(tform,[1;2*RADIUS;2*RADIUS;1;1],[1;1;2*RADIUS;2*RADIUS;1]);
        hold on; plot(i_x,i_y,'r'); hold off;
        
        figure(2*i_rect); clf;
        imagesc(patch); axis equal; axis off;
        [t_x,t_y] = transformPointsForward(tform,annorect.objpos.x,annorect.objpos.y);
        hold on; plot(t_x,t_y,'or'); hold off;
        hold on; plot(pos_xy(1),pos_xy(2),'*r'); hold off;
    end
    
    new_annorect(i_rect).objpos.x = pos_xy(1);
    new_annorect(i_rect).objpos.y = pos_xy(2);
    new_annorect(i_rect).scale = scale;
    new_annorect(i_rect).tform = tform;
    new_annorect(i_rect).image = patch;
end

end