%% DEMO_TRACKING


rootdir = '/home/fan/Desktop/Object_Tracking/Demo/DSiamE/';
seq = struct('name','skiing','path',[rootdir,'sequences/Skiing/'],'startFrame',1,'endFrame',81,'nz',4,'ext','jpg','init_rect', [0,0,0,0]);

seq.len = seq.endFrame - seq.startFrame + 1;
seq.s_frames = cell(seq.len,1);
nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
for i=1:seq.len
    image_no = seq.startFrame + (i-1);
    id = sprintf(nz,image_no);
    seq.s_frames{i} = strcat(seq.path,'img/',id,'.',seq.ext);
end

rect_anno = dlmread([seq.path 'groundtruth_rect.txt']);
seq.init_rect = rect_anno(seq.startFrame,:);
isDisplay = 1;

% the pretrained network for Dynamic Siamese Network 
netname = 'siamfc';
% '1res' denotes the multi-layer DSiam (DSiamM in paper) and uses two layers for tracking
% '0res' denotes the single-layer DSiam (DSiam in paper) and uses the last layer for tracking
nettype = '1res';
run_DSiam(seq,[],isDisplay,rootdir,netname,nettype);