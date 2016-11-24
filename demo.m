function demo()
% Human Pose Estimation demo function

MATCAFFE = '/home/ita/code/caffe4/matlab'; %replace with your caffe matlab folder
TRW_S = '/home/ita/code/TRW_S-v1.3/'; %replace with your TRW-S folder
CAFFE_MODEL = 'HPE-WIS.caffemodel';

if (~exist(MATCAFFE,'dir'))
    error('please install caffe: https://github.com/BVLC/caffe');
else
    addpath(genpath(MATCAFFE))
end

if (~exist(TRW_S,'dir'))
    error('please install TRW-S: https://github.com/markeroon/matlab-computer-vision-routines/blob/master/third_party/imrender/vgg/vgg_trw_bp.m');
else
    addpath(genpath(TRW_S));
end

if (~exist(fullfile('net',CAFFE_MODEL),'file'))
    error('please copy into folder "net" model file: http://www.wisdom.weizmann.ac.il/~vision/deep-voting-hpe/HPE-WIS.caffemodel');
end

imageDir = '.';
imname = '072095531_001.png';

imsize = [504,504];
head_size = 64;

annorects(1).scale = 0.03*head_size;
annorects(1).objpos.x = imsize(2)./2;
annorects(1).objpos.y = imsize(1)./2;

annolist.image.name = imname;
annolist.annorect = annorects;

demo_mpii(imageDir, annolist);

end
