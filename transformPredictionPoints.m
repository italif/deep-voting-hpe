function annorect = transformPredictionPoints(pred_xy,t_annorect)
    
    scale = t_annorect.tform.T(3,3)*t_annorect.scale;
    [objpos.x,objpos.y] = transformPointsInverse(t_annorect.tform,t_annorect.objpos.x,t_annorect.objpos.y);
    for i_pt = 1:size(pred_xy,1)
        [point.x,point.y] = transformPointsInverse(t_annorect.tform,pred_xy(i_pt,1),pred_xy(i_pt,2));
        point.id = i_pt-1;
        point.is_visible = 1;
        annopoints.point(i_pt) = point;
    end
    
    annorect.scale = scale;
    annorect.objpos = objpos;
    annorect.annopoints = annopoints;

end