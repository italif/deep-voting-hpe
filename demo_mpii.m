function annolist = demo_mpii(imageDir, annolist)
% Human Pose Estimation demo function on MPII data structure
% Input:
%   imageDir: input image folder name
%   annolist: structure with the floowing fields:
%       annolist.image.name - image file name
%       annolist.annorect(i).scale - person scale
%       annolist.annorect(i).objpos - person mid location

img = imread([imageDir '/' annolist.image.name]);
annorects = annolist.annorect;

annorects = run_on_image(img, annorects);
annolist.annorect = annorects;

sample.img = img;
sample.annorect = annorects;
show_stick_man(sample);

end
