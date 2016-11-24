
function annorects = transform_back_pred(annorects, new_annorects, predictions)
eps_val = eps('single');
for i_rect = 1:length(annorects)
    if (~isfield(annorects(i_rect), 'scale') || ...
            ~isfield(annorects(i_rect), 'objpos') || ...
            isempty(annorects(i_rect).scale) || ...
            isempty(annorects(i_rect).objpos))
        continue;
    end
    pred = predictions{i_rect};
    annorect = transformPredictionPoints(pred(1:16,2:-1:1),new_annorects(i_rect));
    s = new_annorects(i_rect).tform.T(3,3);
    if (...
            (abs(annorects(i_rect).scale - annorect.scale)>eps_val*s) || ...
            (abs(annorects(i_rect).objpos.x - annorect.objpos.x)>(0.5+eps_val)*s) || ...
            (abs(annorects(i_rect).objpos.y - annorect.objpos.y)>(0.5+eps_val)*s))
        error(['failed on image ' name]);
    end
    reine_points = true;
    if (reine_points)
        d_x = annorects(i_rect).objpos.x - annorect.objpos.x;
        d_y = annorects(i_rect).objpos.y - annorect.objpos.y;
        for i_pt = 1:length(annorect.annopoints.point)
            annorect.annopoints.point(i_pt).x = annorect.annopoints.point(i_pt).x + d_x;
            annorect.annopoints.point(i_pt).y = annorect.annopoints.point(i_pt).y + d_y;
        end
    end
    annorects(i_rect).annopoints=annorect.annopoints;
end
end
