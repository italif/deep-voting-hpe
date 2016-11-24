function h = show_stick_man(sample, show_label, i_fig)

if (~exist('i_fig','var'))
    i_fig = 100;
end

if (~exist('show_label','var'))
    show_label = false;
end

C = get_C();

h = figure(i_fig);
imagesc(sample.img); axis equal; axis off;

for i_rect = 1:length(sample.annorect)
[dense_pts_xy] = get_dense_pts_xy(sample.annorect(i_rect).annopoints.point, C);
pts_xy = dense_pts_xy(1:C.N_pts,:);

stick_start_xy = pts_xy(C.Dense_map(:,1)+1,1:2);
stick_end_xy = pts_xy(C.Dense_map(:,2)+1,1:2);
visible = logical(pts_xy(:,3));

hold on;
%plot([stick_start_xy(:,1), stick_end_xy(:,1)]',[stick_start_xy(:,2), stick_end_xy(:,2)]','LineWidth',2);
stick_color = hsv(size(stick_start_xy,1));
for i_stick = 1:size(stick_start_xy,1)
    plot([stick_start_xy(i_stick,1), stick_end_xy(i_stick,1)]',[stick_start_xy(i_stick,2), stick_end_xy(i_stick,2)]','LineWidth',2,'Color',stick_color(i_stick,:));
end
plot(pts_xy(visible,1), pts_xy(visible,2),'o','MarkerEdgeColor','r','MarkerFaceColor','b');
plot(pts_xy(~visible,1), pts_xy(~visible,2),'o','MarkerEdgeColor','r','MarkerFaceColor','w');
if (show_label)
    labels = strrep(C.Pts_list, '_', '.');
    text(pts_xy(:,1), pts_xy(:,2), labels(1:size(pts_xy,1)), 'Color', 'y', 'VerticalAlignment', 'bottom');
end
hold off;

end

return;

function [pts_xy] = get_pts(annopoints, C)
pt_id = (0:15)';
pt_indx = cell2mat(arrayfun(@(x) [annopoints.id]==x, pt_id, 'UniformOutput', false))*(1:length(annopoints))';

pts_xy = [nan(length(pt_id),2),zeros(length(pt_id),1),pt_id];
is_visible_exist = cellfun(@(x) ~isempty(x), {annopoints.is_visible});
is_visible = ones(1,length(annopoints));
is_visible(is_visible_exist)=[annopoints.is_visible];
pts_xy(pt_id(pt_indx>0)+1,1:3) = [[annopoints(pt_indx(pt_indx>0)).x]',[annopoints(pt_indx(pt_indx>0)).y]',is_visible(pt_indx(pt_indx>0))'];

r_hand_xyv = get_center_hand(pts_xy(C.r_elbow+1,:),pts_xy(C.r_wrist+1,:));
l_hand_xyv = get_center_hand(pts_xy(C.l_elbow+1,:),pts_xy(C.l_wrist+1,:));

pts_xy = [pts_xy; [r_hand_xyv, C.r_hand]; [l_hand_xyv, C.l_hand]];
return;

function pts_xy = get_dense_pts_xy(annopoints,C)
pts_xy = get_pts(annopoints,C);
dense_pts_xy = 0.5*(pts_xy(C.Dense_map(:,2)+1,1:2)+pts_xy(C.Dense_map(:,1)+1,1:2));
dense_pts_xy_visible = (pts_xy(C.Dense_map(:,1)+1,3) & pts_xy(C.Dense_map(:,2)+1,3));
dense_pts_xy_id = (C.N_pts+1:C.N_dense_pts)';
pts_xy = [pts_xy; [dense_pts_xy, dense_pts_xy_visible, dense_pts_xy_id]];
return;

function [hand_xyv] = get_center_hand(elbow_xyv, wrist_xyv)
if (any(isnan([elbow_xyv,wrist_xyv])))
    hand_xyv = [nan,nan,0];
else
    hand_direction = wrist_xyv(1:2)-elbow_xyv(1:2);
    center_hand_xy = wrist_xyv(1:2) + 0.3*hand_direction;
    is_visible = wrist_xyv(3);
    hand_xyv = [center_hand_xy, is_visible];
end

return;